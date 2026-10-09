// Alex LLM API — "Refresh Models" button.
//
// Adds a button widget to the AlexLLMNode. Clicking it POSTs the node's
// current openai_api_key + endpoint to /alex_llm/refresh_models; the backend
// calls GET {endpoint}/models, caches the result, and returns the model list,
// which we then inject into the `model` combo widget live.
//
// On node creation we also sync the combo from the server-side cache
// (GET /alex_llm/models_cache) so refreshes done in another tab/session
// show up without a full page reload.

import { app } from "../../../scripts/app.js";

const NODE_NAME = "AlexLLMNode";
const REFRESH_URL = "/alex_llm/refresh_models";
const CACHE_URL = "/alex_llm/models_cache";

function findWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function applyModels(modelWidget, models) {
  if (!modelWidget || !Array.isArray(models) || models.length === 0) return false;
  if (!modelWidget.options) modelWidget.options = {};
  modelWidget.options.values = models;
  if (!models.includes(modelWidget.value)) {
    modelWidget.value = models[0];
  }
  return true;
}

app.registerExtension({
  name: "Alex.LLM.API.RefreshModels",

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) return;

    const origOnCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      const r = origOnCreated ? origOnCreated.apply(this, arguments) : undefined;
      const node = this;

      const btn = node.addWidget(
        "button",
        "\u{1F504} Refresh Models",
        "alex_llm_refresh",
        async () => {
          const wKey = findWidget(node, "openai_api_key");
          const wEp = findWidget(node, "endpoint");
          const wModel = findWidget(node, "model");

          btn.label = "\u23F3 Refreshing...";
          node.setDirtyCanvas(true, false);

          try {
            const resp = await fetch(REFRESH_URL, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                endpoint: wEp ? wEp.value : "",
                api_key: wKey ? wKey.value : "",
              }),
            });
            const data = await resp.json();

            if (data.ok && applyModels(wModel, data.models)) {
              btn.label = `\u{1F504} Refresh Models (${data.models.length})`;
              if (wModel) wModel.value = wModel.value; // trigger combo re-render
            } else {
              btn.label = `\u26A0 ${(data.error || "no models").slice(0, 60)}`;
            }
          } catch (e) {
            btn.label = `\u26A0 ${String(e.message || e).slice(0, 60)}`;
          }

          node.setDirtyCanvas(true, true);
        }
      );
      btn.serialize = false;

      // Sync combo from the server cache (models cached from previous
      // sessions or other tabs). Silent; keeps the default list on failure.
      (async () => {
        try {
          const resp = await fetch(CACHE_URL);
          const data = await resp.json();
          if (!data.ok || !data.cache) return;

          const wEp = findWidget(node, "endpoint");
          const wModel = findWidget(node, "model");
          const ep = (wEp && wEp.value ? wEp.value : "").trim().replace(/\/+$/, "");

          let models = [];
          if (ep && Array.isArray(data.cache[ep])) {
            models = data.cache[ep];
          } else {
            // union of all cached endpoints
            for (const list of Object.values(data.cache)) models = models.concat(list);
            models = [...new Set(models)].sort();
          }
          if (applyModels(wModel, models)) {
            btn.label = `\u{1F504} Refresh Models (${models.length})`;
            node.setDirtyCanvas(true, false);
          }
        } catch (e) {
          /* keep defaults */
        }
      })();

      return r;
    };
  },
});
