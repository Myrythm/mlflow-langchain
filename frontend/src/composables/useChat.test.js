import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api", () => ({ streamChat: vi.fn() }));

import { streamChat } from "../api";
import { useChat } from "./useChat";

const SETTINGS = { mode: "hybrid", sparse_model: "bm25", k: 4 };
const SOURCES = [{ title: "Pricing", category: "billing", snippet: "s" }];

beforeEach(() => {
  streamChat.mockReset();
});

function happyStream() {
  streamChat.mockImplementation(async (payload, cb) => {
    cb.onSources(SOURCES);
    cb.onToken("A");
    cb.onToken("B");
    cb.onDone();
  });
}

describe("useChat.send", () => {
  it("appends the turn, stores sources and streamed content", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "How much is Pro?";

    await chat.send(SETTINGS);

    expect(chat.messages.value.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(chat.messages.value[1].content).toBe("AB");
    expect(chat.messages.value[1].sources).toEqual(SOURCES);
    expect(chat.stage.value).toBe("idle");
    expect(chat.draft.value).toBe("");
  });

  it("sends prior turns as history and the retrieval settings", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "first";
    await chat.send(SETTINGS);
    chat.draft.value = "second";

    await chat.send({ mode: "dense", sparse_model: "bm25", k: 2 });

    const payload = streamChat.mock.calls[1][0];
    expect(payload.history).toEqual([
      { role: "user", content: "first" },
      { role: "assistant", content: "AB" },
    ]);
    expect(payload).toMatchObject({ question: "second", mode: "dense", k: 2 });
  });

  it("ignores blank drafts and does not call the API", async () => {
    const chat = useChat();
    chat.draft.value = "   ";
    await chat.send(SETTINGS);
    expect(streamChat).not.toHaveBeenCalled();
    expect(chat.messages.value).toEqual([]);
  });

  it("replaces a partial answer with the error and restores the draft", async () => {
    streamChat.mockImplementation(async (payload, cb) => {
      cb.onSources(SOURCES);
      cb.onToken("partial ");
      cb.onError("stream broke politely");
    });
    const chat = useChat();
    chat.draft.value = "hello";

    await chat.send(SETTINGS);

    const reply = chat.messages.value[1];
    expect(reply.error).toBe(true);
    expect(reply.content).toBe("stream broke politely");
    expect(reply.content).not.toContain("partial");
    expect(chat.draft.value).toBe("hello");
    expect(chat.stage.value).toBe("idle");
  });

  it("turns a rejected request (pre-stream HTTP error) into an error turn", async () => {
    streamChat.mockRejectedValue(new Error("OPENAI_API_KEY is missing"));
    const chat = useChat();
    chat.draft.value = "hello";

    await chat.send(SETTINGS);

    expect(chat.messages.value[1].error).toBe(true);
    expect(chat.messages.value[1].content).toContain("OPENAI_API_KEY");
    expect(chat.draft.value).toBe("hello");
  });

  it("excludes error turns from subsequent history", async () => {
    streamChat.mockRejectedValueOnce(new Error("boom"));
    const chat = useChat();
    chat.draft.value = "hello";
    await chat.send(SETTINGS);

    happyStream();
    chat.draft.value = "hello again";
    await chat.send(SETTINGS);

    const payload = streamChat.mock.calls[1][0];
    expect(payload.history).toEqual([{ role: "user", content: "hello" }]);
  });
});

describe("stage + sources selection", () => {
  it("walks idle -> retrieving -> generating -> idle", async () => {
    const seen = [];
    const chat = useChat();
    streamChat.mockImplementation(async (payload, cb) => {
      seen.push(chat.stage.value); // retrieving
      cb.onSources(SOURCES);
      seen.push(chat.stage.value); // generating
      cb.onDone();
    });
    chat.draft.value = "q";

    await chat.send(SETTINGS);

    expect(seen).toEqual(["retrieving", "generating"]);
    expect(chat.stage.value).toBe("idle");
  });

  it("activeSources follows the latest turn unless one is selected", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "q1";
    await chat.send(SETTINGS);
    expect(chat.activeSources.value).toEqual(SOURCES);

    chat.selectedIndex.value = 0; // a user message -> no sources
    expect(chat.activeSources.value).toEqual([]);
  });

  it("clear resets everything", async () => {
    happyStream();
    const chat = useChat();
    chat.draft.value = "q";
    await chat.send(SETTINGS);

    chat.clear();

    expect(chat.messages.value).toEqual([]);
    expect(chat.stage.value).toBe("idle");
    expect(chat.selectedIndex.value).toBe(null);
  });
});
