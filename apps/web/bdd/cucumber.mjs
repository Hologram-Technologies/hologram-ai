// cucumber-js profiles for the browser-executor rows. The feature files live
// in the repo-root features/ tree (the same dictionary-governed set the Rust
// runner uses); this runner takes the @executor:browser slice.
const common = {
  paths: ["../../features/suites/**/*.feature"],
  import: ["bdd/world.mjs", "bdd/steps.mjs"],
  strict: true, // undefined/pending steps fail — the no-skip discipline
  format: ["progress"],
};

export default {
  ...common,
  tags: "@executor:browser and not @live",
};

export const live = {
  ...common,
  tags: "@executor:browser and @live",
};
