<script setup>
import RetrievalControls from "./RetrievalControls.vue";
import SourceCard from "./SourceCard.vue";

defineProps({
  config: { type: Object, default: null },
  settings: { type: Object, required: true },
  sources: { type: Array, default: () => [] },
});
defineEmits(["update:settings"]);
</script>

<template>
  <aside class="flex min-h-0 flex-col overflow-y-auto border-l border-line">
    <section class="border-b border-line p-4">
      <h2 class="font-mono text-[11px] tracking-widest text-muted uppercase">retrieval</h2>
      <RetrievalControls
        v-if="config"
        class="mt-3"
        :config="config"
        :settings="settings"
        @update:settings="$emit('update:settings', $event)"
      />
      <p v-else class="mt-3 font-mono text-xs text-muted">backend unreachable</p>
    </section>

    <section class="p-4">
      <h2 class="font-mono text-[11px] tracking-widest text-muted uppercase">
        sources<span v-if="sources.length"> · {{ sources.length }}</span>
      </h2>
      <p v-if="!sources.length" class="mt-3 text-sm text-muted">
        Ask a question to see which knowledge-base passages the answer is grounded in.
      </p>
      <ol v-else class="mt-3">
        <SourceCard v-for="(src, i) in sources" :key="i" :index="i + 1" :source="src" />
      </ol>
    </section>
  </aside>
</template>
