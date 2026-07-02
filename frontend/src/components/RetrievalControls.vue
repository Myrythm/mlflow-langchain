<script setup>
const props = defineProps({
  config: { type: Object, required: true },
  settings: { type: Object, required: true },
});
const emit = defineEmits(["update:settings"]);

function set(key, value) {
  emit("update:settings", { ...props.settings, [key]: value });
}
</script>

<template>
  <div class="space-y-3 font-mono text-xs">
    <label class="block">
      <span class="text-muted">mode</span>
      <select
        :value="settings.mode"
        class="mt-1 w-full border border-line bg-transparent px-2 py-1.5
          focus:border-ink focus:outline-none"
        @change="set('mode', $event.target.value)"
      >
        <option v-for="m in config.modes" :key="m" :value="m">{{ m }}</option>
      </select>
    </label>

    <label class="block">
      <span class="text-muted">sparse model</span>
      <select
        :value="settings.sparse_model"
        class="mt-1 w-full border border-line bg-transparent px-2 py-1.5
          focus:border-ink focus:outline-none"
        @change="set('sparse_model', $event.target.value)"
      >
        <option v-for="m in config.sparse_models" :key="m" :value="m">{{ m }}</option>
      </select>
    </label>

    <label class="block">
      <span class="text-muted">top-k · {{ settings.k }}</span>
      <input
        type="range"
        :min="config.k_min"
        :max="config.k_max"
        :value="settings.k"
        class="mt-1 w-full accent-accent"
        @input="set('k', Number($event.target.value))"
      />
    </label>
  </div>
</template>
