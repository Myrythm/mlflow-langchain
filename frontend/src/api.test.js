import { describe, expect, it, vi } from "vitest";

import { createSSEParser } from "./api";

function collect() {
  const events = [];
  const feed = createSSEParser((event, data) => events.push([event, data]));
  return { events, feed };
}

describe("createSSEParser", () => {
  it("parses a complete event", () => {
    const { events, feed } = collect();
    feed('event: token\ndata: {"text":"A"}\n\n');
    expect(events).toEqual([["token", '{"text":"A"}']]);
  });

  it("handles events split across chunk boundaries", () => {
    const { events, feed } = collect();
    feed("event: tok");
    feed('en\ndata: {"te');
    feed('xt":"AB"}\n\n');
    expect(events).toEqual([["token", '{"text":"AB"}']]);
  });

  it("handles multiple events in one chunk, in order", () => {
    const { events, feed } = collect();
    feed(
      'event: sources\ndata: []\n\nevent: token\ndata: {"text":"A"}\n\nevent: done\ndata: {}\n\n',
    );
    expect(events.map(([e]) => e)).toEqual(["sources", "token", "done"]);
  });

  it("ignores comments and blocks without data", () => {
    const { events, feed } = collect();
    feed(": ping\n\nevent: ghost\n\ndata: {}\n\n");
    expect(events).toEqual([["message", "{}"]]);
  });
});

describe("streamChat", () => {
  it("routes events to the right callbacks", async () => {
    const body = [
      'event: sources\ndata: [{"title":"Pricing","category":"billing","snippet":"s"}]\n\n',
      'event: token\ndata: {"text":"A"}\n\nevent: token\ndata: {"text":"B"}\n\n',
      "event: done\ndata: {}\n\n",
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        body: new ReadableStream({
          start(controller) {
            const enc = new TextEncoder();
            body.forEach((chunk) => controller.enqueue(enc.encode(chunk)));
            controller.close();
          },
        }),
      }),
    );
    const { streamChat } = await import("./api");
    const got = { sources: null, text: "", done: false };

    await streamChat(
      { question: "q" },
      {
        onSources: (s) => (got.sources = s),
        onToken: (t) => (got.text += t),
        onError: () => {},
        onDone: () => (got.done = true),
      },
    );

    expect(got.sources).toEqual([
      { title: "Pricing", category: "billing", snippet: "s" },
    ]);
    expect(got.text).toBe("AB");
    expect(got.done).toBe(true);
    vi.unstubAllGlobals();
  });

  it("throws the backend detail message on a non-2xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({ detail: "OPENAI_API_KEY is missing" }),
      }),
    );
    const { streamChat } = await import("./api");

    await expect(
      streamChat({ question: "q" }, { onSources() {}, onToken() {}, onError() {}, onDone() {} }),
    ).rejects.toThrow("OPENAI_API_KEY is missing");
    vi.unstubAllGlobals();
  });
});
