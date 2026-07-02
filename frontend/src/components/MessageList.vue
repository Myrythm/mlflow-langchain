<script setup>
import { nextTick, ref, watch } from "vue";

import Message from "./Message.vue";
import StatusLine from "./StatusLine.vue";

const props = defineProps({
  messages: { type: Array, required: true },
  stage: { type: String, required: true },
  selectedIndex: { type: Number, default: null },
  examples: { type: Array, default: () => [] },
});
defineEmits(["select", "example"]);

const scroller = ref(null);
watch(
  () => [props.messages.length, props.messages.at(-1)?.content],
  async () => {
    await nextTick();
    scroller.value?.scrollTo({ top: scroller.value.scrollHeight });
  },
);
</script>

<template>
  <div ref="scroller" class="overflow-y-auto">
    <div class="mx-auto w-full max-w-[65ch] px-6 pt-4 pb-10">
      <div v-if="messages.length === 0" class="pt-16">
        <p class="font-mono text-[11px] tracking-widest text-muted uppercase">
          try one of these
        </p>
        <ul class="mt-4 divide-y divide-line border-y border-line">
          <li v-for="q in examples" :key="q">
            <button
              class="w-full py-3 text-left text-[15px] hover:text-accent"
              @click="$emit('example', q)"
            >
              {{ q }}
            </button>
          </li>
        </ul>
      </div>

      <template v-for="(msg, i) in messages" :key="i">
        <Message
          :message="msg"
          :selected="i === selectedIndex"
          :streaming="i === messages.length - 1 && stage === 'generating'"
          @select="$emit('select', i)"
        />
        <StatusLine
          v-if="i === messages.length - 1 && msg.role === 'assistant' && !msg.error"
          :stage="stage"
          :timings="msg.timings ?? {}"
        />
      </template>
    </div>
  </div>
</template>
