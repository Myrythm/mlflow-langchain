// HTTP client for the FastAPI backend. The Vite dev server proxies /api to :8000.

async function getJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json();
}

export function fetchConfig() {
  return getJSON("/api/config");
}

export function fetchHealth() {
  return getJSON("/api/health");
}

// Incremental text/event-stream parser. Returns feed(chunk); calls
// onEvent(eventName, rawData) once per complete SSE message. Buffers partial
// messages across chunks, ignores comment lines (":") and messages without data.
export function createSSEParser(onEvent) {
  let buffer = "";
  return function feed(chunk) {
    buffer += chunk;
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      let event = "message";
      const dataLines = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length) onEvent(event, dataLines.join("\n"));
    }
  };
}

export async function streamChat(payload, { onSources, onToken, onError, onDone }) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let detail = `chat request failed: ${res.status}`;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // keep the generic message when the error body is not JSON
    }
    throw new Error(detail);
  }

  const feed = createSSEParser((event, data) => {
    if (event === "sources") onSources(JSON.parse(data));
    else if (event === "token") onToken(JSON.parse(data).text);
    else if (event === "error") onError(JSON.parse(data).message);
    else if (event === "done") onDone();
  });
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    feed(decoder.decode(value, { stream: true }));
  }
}
