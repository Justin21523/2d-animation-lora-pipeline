async function loadManifest() {
  const response = await fetch("demo-data/manifest.json");
  if (!response.ok) {
    throw new Error(`Manifest request failed: ${response.status}`);
  }
  return response.json();
}

function setText(id, value) {
  const element = document.getElementById(id);
  if (element) element.textContent = value;
}

function renderSummary(data) {
  setText("stages-complete", `${data.summary.stages_complete}/${data.summary.stages_total}`);
  setText("metadata-rows", data.summary.metadata_rows.toLocaleString());
  setText("image-artifacts", data.summary.image_artifacts.toLocaleString());
  setText("demo-status", data.summary.demo_ready ? "Demo ready" : "Needs attention");
  setText("results-headline", data.product_results.headline);
}

function renderStages(data) {
  const grid = document.getElementById("stage-grid");
  grid.innerHTML = data.stages
    .map(
      (stage) => `
        <article class="stage-card">
          <span class="stage-order">${stage.order}</span>
          <strong>${stage.label}</strong>
          <p>${stage.description}</p>
          <dl>
            <div><dt>Rows</dt><dd>${stage.rows}</dd></div>
            <div><dt>Assets</dt><dd>${stage.images}</dd></div>
          </dl>
        </article>
      `,
    )
    .join("");
}

function renderResults(data) {
  const assets = data.product_results.assets;
  document.getElementById("before-after-image").src = assets.before_after;
  document.getElementById("video-source").src = data.product_results.media.video;
  document.getElementById("video-player").poster = data.product_results.media.poster;
  document.getElementById("download-video").href = data.product_results.media.video;
  document.getElementById("cover-image").src = assets.character_sheet;

  document.getElementById("metric-grid").innerHTML = data.product_results.metrics
    .map(
      (metric) => `
        <article>
          <span>${metric.value}</span>
          <strong>${metric.label}</strong>
          <small>${metric.trend}</small>
        </article>
      `,
    )
    .join("");

  const galleryItems = [
    ["Character dataset sheet", assets.character_sheet],
    ["Training metrics", assets.training_metrics],
    ["Evaluation matrix", assets.evaluation_matrix],
    ["Motion strip", assets.animation_strip],
  ];
  document.getElementById("result-gallery").innerHTML = galleryItems
    .map(
      ([label, src]) => `
        <figure>
          <img src="${src}" alt="${label}" loading="lazy" />
          <figcaption>${label}</figcaption>
        </figure>
      `,
    )
    .join("");

  document.getElementById("screenshot-grid").innerHTML = data.product_results.media.screenshots
    .map(
      (src) => `
        <figure>
          <img src="${src}" alt="2D Animation LoRA Pipeline screenshot" loading="lazy" />
          <figcaption>${src.split("/").pop()}</figcaption>
        </figure>
      `,
    )
    .join("");

  document.getElementById("deliverable-grid").innerHTML = data.product_results.deliverables
    .map((item) => `<article><span>✓</span><strong>${item}</strong></article>`)
    .join("");
}

function renderScenarios(data) {
  document.getElementById("scenario-list").innerHTML = data.scenarios
    .map(
      (scenario) => `
        <article>
          <h3>${scenario.title}</h3>
          <p>${scenario.description}</p>
        </article>
      `,
    )
    .join("");
}

loadManifest()
  .then((data) => {
    renderSummary(data);
    renderStages(data);
    renderResults(data);
    renderScenarios(data);
    document.getElementById("video-player").load();
  })
  .catch((error) => {
    document.body.classList.add("manifest-error");
    console.error(error);
  });
