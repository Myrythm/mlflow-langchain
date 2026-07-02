<script setup>
defineProps({
  index: { type: Number, required: true },
  source: { type: Object, required: true },
});
</script>

<template>
  <li class="border-t border-line pt-3 first:border-t-0 first:pt-0">
    <div class="flex items-baseline gap-2">
      <span class="font-mono text-xs font-medium text-accent">
        {{ String(index).padStart(2, "0") }}
      </span>
      <span class="text-sm font-medium">{{ source.title }}</span>
    </div>
    <span class="mt-0.5 inline-block font-mono text-[11px] text-muted">
      {{ source.category }}
    </span>
    <p class="mt-1 text-[13px] leading-relaxed text-muted">{{ source.snippet }}</p>
  </li>
</template>
