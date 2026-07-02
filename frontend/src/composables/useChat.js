import { computed, ref } from "vue";

import { streamChat } from "../api";

const ERROR_REPLY =
  "Sorry — something went wrong while generating an answer. Please try again in a " +
  "moment; your question is still in the input box.";

// Conversation state for one chat session. Stateless server: prior turns are
// replayed as `history` on every request (error turns excluded).
export function useChat() {
  const messages = ref([]);
  const stage = ref("idle"); // idle | retrieving | generating
  const draft = ref("");
  const selectedIndex = ref(null);

  const activeSources = computed(() => {
    const pick =
      selectedIndex.value !== null
        ? messages.value[selectedIndex.value]
        : [...messages.value].reverse().find((m) => m.role === "assistant");
    return pick?.sources ?? [];
  });

  function clear() {
    messages.value = [];
    stage.value = "idle";
    selectedIndex.value = null;
  }

  async function send(settings) {
    const question = draft.value.trim();
    if (!question || stage.value !== "idle") return;

    const history = messages.value
      .filter((m) => !m.error)
      .map(({ role, content }) => ({ role, content }));

    draft.value = "";
    selectedIndex.value = null;
    messages.value.push({ role: "user", content: question });
    messages.value.push({ role: "assistant", content: "", sources: [], timings: {} });
    const reply = messages.value[messages.value.length - 1];

    stage.value = "retrieving";
    const t0 = performance.now();

    const fail = (message) => {
      reply.content = message || ERROR_REPLY;
      reply.error = true;
      draft.value = question; // easy retry, like the old UI
      stage.value = "idle";
    };

    try {
      await streamChat(
        { question, history, ...settings },
        {
          onSources(sources) {
            reply.sources = sources;
            reply.timings.retrieval = Math.round(performance.now() - t0);
            stage.value = "generating";
          },
          onToken(text) {
            reply.content += text;
          },
          onError(message) {
            fail(message);
          },
          onDone() {
            reply.timings.generation =
              Math.round(performance.now() - t0) - (reply.timings.retrieval ?? 0);
            stage.value = "idle";
          },
        },
      );
    } catch (err) {
      fail(err.message);
    }
    if (stage.value !== "idle") stage.value = "idle"; // stream ended without done/error
  }

  return { messages, stage, draft, selectedIndex, activeSources, send, clear };
}
