<script setup>
defineProps({
  stage: { type: String, required: true },
  timings: { type: Object, default: () => ({}) },
});
</script>

<template>
  <p class="mt-2 flex gap-4 font-mono text-[11px] text-muted">
    <span v-if="stage === 'retrieving'" class="text-accent">retrieving…</span>
    <span v-else-if="timings.retrieval != null">retrieving · {{ timings.retrieval }}ms</span>
    <span v-if="stage === 'generating'" class="text-accent">generating…</span>
    <span v-else-if="timings.generation != null">generating · {{ timings.generation }}ms</span>
  </p>
</template>
