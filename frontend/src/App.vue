<script setup>
import { onMounted, ref } from "vue";

import { fetchConfig, fetchHealth } from "./api";
import Composer from "./components/Composer.vue";
import InspectorPanel from "./components/InspectorPanel.vue";
import MessageList from "./components/MessageList.vue";
import { useChat } from "./composables/useChat";

const { messages, stage, draft, selectedIndex, activeSources, send, clear } = useChat();

const config = ref(null);
const health = ref(null);
const settings = ref({ mode: "hybrid", sparse_model: "bm25", k: 4 });

function ask(question) {
  draft.value = question;
  send(settings.value);
}

onMounted(async () => {
  try {
    config.value = await fetchConfig();
    settings.value = {
      mode: config.value.default_mode,
      sparse_model: config.value.default_sparse,
      k: config.value.default_k,
    };
  } catch {
    config.value = null; // inspector shows "backend unreachable"
  }
  try {
    health.value = await fetchHealth();
  } catch {
    health.value = null;
  }
});
</script>

<template>
  <div class="flex h-full flex-col">
    <header
      class="flex h-11 shrink-0 items-center justify-between border-b border-line px-4"
    >
      <span class="font-mono text-sm font-medium">
        nimbus<span class="text-accent">/</span>support
      </span>
      <span class="flex items-center gap-2 font-mono text-[11px] text-muted">
        <span
          class="inline-block h-2 w-2 rounded-full"
          :class="health?.status === 'ok' ? 'bg-accent' : 'bg-line'"
        ></span>
        {{
          health === null
            ? "backend offline"
            : health.mlflow_reachable
              ? "api + mlflow"
              : "api up · mlflow down"
        }}
      </span>
    </header>

    <div class="grid min-h-0 flex-1 grid-cols-[minmax(0,1fr)_320px]">
      <main class="flex min-h-0 flex-col">
        <MessageList
          class="min-h-0 flex-1"
          :messages="messages"
          :stage="stage"
          :selected-index="selectedIndex"
          :examples="config?.example_questions ?? []"
          @select="selectedIndex = $event"
          @example="ask"
        />
        <Composer
          v-model="draft"
          :busy="stage !== 'idle'"
          @send="send(settings)"
          @clear="clear"
        />
      </main>
      <InspectorPanel
        :config="config"
        :settings="settings"
        :sources="activeSources"
        @update:settings="settings = $event"
      />
    </div>
  </div>
</template>
