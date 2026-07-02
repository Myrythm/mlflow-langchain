<script setup>
import DOMPurify from "dompurify";
import { marked } from "marked";
import { computed } from "vue";

const props = defineProps({
  message: { type: Object, required: true },
  selected: { type: Boolean, default: false },
  streaming: { type: Boolean, default: false },
});
defineEmits(["select"]);

const html = computed(() =>
  DOMPurify.sanitize(marked.parse(props.message.content ?? "")),
);
</script>

<template>
  <div v-if="message.role === 'user'" class="mt-10 flex flex-col items-end">
    <span class="font-mono text-[11px] tracking-widest text-muted uppercase">you</span>
    <p class="mt-1 max-w-[52ch] text-right text-[15px] font-medium">
      {{ message.content }}
    </p>
  </div>

  <div v-else class="mt-4">
    <button
      class="font-mono text-[11px] tracking-widest uppercase"
      :class="selected ? 'text-accent' : 'text-muted hover:text-accent'"
      @click="$emit('select')"
    >
      agent<span v-if="message.sources?.length"> · {{ message.sources.length }} src</span>
    </button>
    <p
      v-if="message.error"
      class="mt-1 border-l-2 border-accent pl-3 text-[15px] text-muted"
    >
      {{ message.content }}
    </p>
    <div v-else class="answer mt-1 text-[15px]" :class="{ streaming }" v-html="html"></div>
  </div>
</template>
