<script setup>
const props = defineProps({
  modelValue: { type: String, required: true },
  busy: { type: Boolean, default: false },
});
const emit = defineEmits(["update:modelValue", "send", "clear"]);

function onKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!props.busy) emit("send");
  }
}
</script>

<template>
  <div class="shrink-0 border-t border-line">
    <div class="mx-auto w-full max-w-[65ch] px-6 py-4">
      <div class="flex items-end gap-3">
        <textarea
          :value="modelValue"
          rows="2"
          placeholder="e.g. How much does the Pro plan cost?"
          class="min-h-[3.25rem] flex-1 resize-none border border-line bg-transparent px-3
            py-2 text-[15px] placeholder:text-muted focus:border-ink focus:outline-none"
          @input="$emit('update:modelValue', $event.target.value)"
          @keydown="onKeydown"
        ></textarea>
        <button
          class="h-[3.25rem] bg-accent px-5 font-mono text-sm font-medium text-paper
            disabled:opacity-40"
          :disabled="busy || !modelValue.trim()"
          @click="$emit('send')"
        >
          send
        </button>
      </div>
      <button
        class="mt-2 font-mono text-[11px] tracking-widest text-muted uppercase
          hover:text-accent disabled:opacity-40"
        :disabled="busy"
        @click="$emit('clear')"
      >
        clear conversation
      </button>
    </div>
  </div>
</template>
