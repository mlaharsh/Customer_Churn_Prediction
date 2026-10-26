/* UI glue for the static Churn Signal scorer. */
(function () {
  "use strict";

  const model = window.CHURN_MODEL;
  const scorer = window.ChurnScorer;

  const $ = (selector, root) => (root || document).querySelector(selector);
  const $$ = (selector, root) =>
    Array.from((root || document).querySelectorAll(selector));

  const state = {
    batchRows: null,
    batchFileName: "",
    threshold: 0,
    currentRaw: null,
  };

  const LAYOUT = [
    {
      title: "Account & billing",
      note: "Spend and commitment",
      fields: [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "PaperlessBilling",
      ],
    },
    {
      title: "Services",
      note: "Plan and add-ons",
      fields: [
        "InternetService",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
      ],
    },
    {
      title: "Customer context",
      note: "Account profile",
      fields: ["gender", "SeniorCitizen", "Partner", "Dependents"],
    },
  ];

  const NUMERIC_FIELDS = {
    tenure: {
      min: 0,
      max: 72,
      step: 1,
      unit: "months",
      help: "Months with the company.",
    },
    MonthlyCharges: {
      min: 0,
      max: 300,
      step: 0.05,
      unit: "USD / month",
      help: "Current recurring charge.",
    },
    TotalCharges: {
      min: 0,
      max: 20000,
      step: 0.01,
      unit: "USD",
      help: "Auto-calculated from tenure × monthly unless you turn that off.",
      wide: true,
    },
  };

  const numericMeta = new Map((model ? model.numeric : []).map((f) => [f.name, f]));
  const categoricalMeta = new Map(
    (model ? model.categorical : []).map((f) => [f.name, f])
  );

  function setText(selector, value) {
    $$(selector).forEach((element) => {
      element.textContent = value;
    });
  }

  function formatDate(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return "trained model";
    }
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  }

  function hydrateMetadata() {
    if (!model) {
      return;
    }
    const metrics = model.metadata.metrics;
    const holdout = scorer.thresholdMetrics(model, model.metadata.threshold);
    setText(".js-threshold", model.metadata.threshold.toFixed(3));
    setText(".js-roc", metrics.rocAuc.toFixed(4));
    setText(".js-model-name", model.metadata.selectedModel);
    setText(".js-flagged", holdout.contacts.toLocaleString("en-US"));
    setText(".js-holdout-n", model.holdout.n.toLocaleString("en-US"));
    setText(
      ".js-caught",
      `${holdout.tp.toLocaleString("en-US")} / ${model.holdout.churners.toLocaleString("en-US")}`
    );
    setText(".js-alert-rate", scorer.formatPct(holdout.alertRate));
    setText(".js-recall", scorer.formatPct(holdout.recall));
    setText(".js-precision", scorer.formatPct(holdout.precision));
    $$(".js-alert-marker").forEach((marker) => {
      marker.style.left = `${(holdout.alertRate * 100).toFixed(2)}%`;
    });
    const stamp = $("#modelStamp");
    if (stamp) {
      stamp.textContent = `${model.metadata.selectedModel} · ${formatDate(
        model.metadata.trainedAt
      )}`;
      stamp.setAttribute("datetime", model.metadata.trainedAt || "");
    }
  }

  function buildField(name) {
    const field = document.createElement("div");
    field.className = "field";
    const numeric = numericMeta.get(name);
    const categorical = categoricalMeta.get(name);
    const meta = numeric || categorical;
    if (!meta) {
      return field;
    }
    const config = NUMERIC_FIELDS[name];
    if (config && config.wide) {
      field.classList.add("field-wide");
    }

    const label = document.createElement("label");
    label.setAttribute("for", `field-${name}`);
    label.appendChild(document.createTextNode(meta.label));
    if (config && config.unit) {
      const unit = document.createElement("span");
      unit.className = "unit";
      unit.textContent = config.unit;
      label.appendChild(unit);
    }
    field.appendChild(label);

    if (numeric) {
      const input = document.createElement("input");
      input.className = "input";
      input.type = "number";
      input.id = `field-${name}`;
      input.name = name;
      input.min = String(config.min);
      input.max = String(config.max);
      input.step = String(config.step);
      input.inputMode = "decimal";
      input.required = true;
      field.appendChild(input);
      if (config.help) {
        const help = document.createElement("p");
        help.className = "field-help";
        help.textContent = config.help;
        field.appendChild(help);
      }
      if (name === "TotalCharges") {
        const auto = document.createElement("label");
        auto.className = "auto-row";
        auto.htmlFor = "autoTotal";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.id = "autoTotal";
        checkbox.checked = true;
        auto.appendChild(checkbox);
        auto.appendChild(
          document.createTextNode("Auto-calculate from tenure × monthly charges")
        );
        field.appendChild(auto);
      }
    } else if (categorical) {
      const select = document.createElement("select");
      select.className = "select";
      select.id = `field-${name}`;
      select.name = name;
      select.required = true;
      Object.keys(categorical.coefficients).forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
      field.appendChild(select);
    }
    return field;
  }

  function renderForm() {
    const body = $("#formBody");
    if (!body) {
      return;
    }
    body.replaceChildren();
    LAYOUT.forEach((group) => {
      const section = document.createElement("section");
      section.className = "form-group";
      const head = document.createElement("div");
      head.className = "form-group-head";
      const title = document.createElement("h4");
      title.textContent = group.title;
      const note = document.createElement("span");
      note.textContent = group.note;
      head.append(title, note);
      const grid = document.createElement("div");
      grid.className = "field-grid";
      group.fields.forEach((name) => {
        grid.appendChild(buildField(name));
      });
      section.append(head, grid);
      body.appendChild(section);
    });
  }

  function exampleById(id) {
    return model.examples.find((example) => example.id === id) || model.examples[0];
  }

  function setFormValues(raw) {
    const autoTotal = $("#autoTotal");
    if (autoTotal) {
      autoTotal.checked = true;
    }
    for (const name of scorer.RAW_FEATURE_COLUMNS) {
      const input = $(`#field-${name}`);
      if (!input) {
        continue;
      }
      if (input.tagName === "SELECT") {
        const value = String(raw[name] == null ? "" : raw[name]);
        if (!Array.from(input.options).some((option) => option.value === value)) {
          input.add(new Option(value, value));
        }
        input.value = value;
      } else {
        input.value = raw[name] == null ? "" : String(raw[name]);
      }
    }
    updateTotalCharges();
    updateScore();
  }

  function loadExample(id) {
    const example = exampleById(id);
    if (example) {
      setFormValues(example.raw);
    }
  }

  function numericInputValue(name) {
    const input = $(`#field-${name}`);
    if (!input) {
      return NaN;
    }
    const value = input.value.trim();
    return value === "" ? NaN : Number(value);
  }

  function updateTotalCharges() {
    const auto = $("#autoTotal");
    const total = $("#field-TotalCharges");
    if (!auto || !total) {
      return;
    }
    const tenure = numericInputValue("tenure");
    const monthly = numericInputValue("MonthlyCharges");
    total.disabled = auto.checked;
    if (auto.checked && Number.isFinite(tenure) && Number.isFinite(monthly)) {
      total.value = String(scorer.roundTo(tenure * monthly, 2));
    }
  }

  function readFormValues() {
    const raw = {};
    for (const name of scorer.RAW_FEATURE_COLUMNS) {
      const input = $(`#field-${name}`);
      if (!input) {
        continue;
      }
      raw[name] = input.value;
    }
    if ($("#autoTotal") && $("#autoTotal").checked) {
      const tenure = numericInputValue("tenure");
      const monthly = numericInputValue("MonthlyCharges");
      if (Number.isFinite(tenure) && Number.isFinite(monthly)) {
        raw.TotalCharges = scorer.roundTo(tenure * monthly, 2);
      }
    }
    return raw;
  }

  function hideError() {
    const box = $("#resultError");
    if (box) {
      box.hidden = true;
      box.textContent = "";
    }
  }

  function showError(message) {
    const box = $("#resultError");
    if (box) {
      box.hidden = false;
      box.textContent = message;
    }
  }

  function renderResult(raw, result) {
    const card = $("#resultCard");
    if (card) {
      card.style.setProperty("--risk", result.riskColor);
    }
    const probability = $("#probabilityValue");
    if (probability) {
      probability.replaceChildren(
        document.createTextNode((result.rawProbability * 100).toFixed(1)),
        (() => {
          const small = document.createElement("small");
          small.textContent = "%";
          return small;
        })()
      );
    }
    const badge = $("#riskBadge");
    if (badge) {
      badge.textContent = `${result.riskTier} risk`;
    }

    const marker = $("#gaugeMarker");
    if (marker) {
      marker.style.left = `${scorer.clamp(result.rawProbability * 100, 0.6, 99.4).toFixed(2)}%`;
    }
    const gauge = $("#riskGauge");
    if (gauge) {
      gauge.setAttribute(
        "aria-label",
        `Churn probability ${(result.rawProbability * 100).toFixed(
          1
        )} percent, ${result.riskTier} risk`
      );
    }

    const decision = $("#decisionText");
    if (decision) {
      decision.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = result.flagged
        ? "Flagged for retention outreach."
        : "Below the outreach threshold.";
      const text = document.createTextNode(
        result.flagged
          ? ` The score is at or above the ${model.metadata.threshold.toFixed(
              3
            )} policy threshold.`
          : ` The score is below the ${model.metadata.threshold.toFixed(
              3
            )} policy threshold; keep standard engagement.`
      );
      decision.append(strong, text);
    }

    const monthly = $("#monthlyRisk");
    const annual = $("#annualRisk");
    if (monthly) {
      monthly.textContent = scorer.formatMoney(
        result.expectedMonthlyRevenueAtRisk
      );
    }
    if (annual) {
      annual.textContent = scorer.formatMoney(
        result.expectedAnnualRevenueAtRisk
      );
    }

    const list = $("#actionList");
    if (list) {
      list.replaceChildren();
      const actions = scorer.retentionActions(
        result.features,
        result.rawProbability,
        model
      );
      actions.forEach((action) => {
        const item = document.createElement("li");
        item.textContent = action;
        list.appendChild(item);
      });
    }

    const drivers = $("#driverList");
    if (drivers) {
      drivers.replaceChildren();
      const rows = scorer.whatIf(result.features, model, 5);
      const maximum = Math.max(
        0.1,
        ...rows.map((row) => Math.abs(row.impactPp))
      );
      rows.forEach((row) => {
        const wrapper = document.createElement("div");
        wrapper.className = "driver";
        const label = document.createElement("span");
        label.className = "driver-label";
        label.textContent = row.label;
        const value = document.createElement("span");
        value.className = "driver-value";
        value.textContent = `${row.current} → ${row.baseline} · ${scorer.formatPp(
          row.impactPp
        )}`;
        const bar = document.createElement("span");
        bar.className = "driver-bar";
        const fill = document.createElement("span");
        fill.className = row.impactPp < 0 ? "negative" : "positive";
        fill.style.width = `${Math.max(
          4,
          (Math.abs(row.impactPp) / maximum) * 100
        ).toFixed(1)}%`;
        bar.appendChild(fill);
        wrapper.append(label, value, bar);
        drivers.appendChild(wrapper);
      });
    }

    hideError();
    state.currentRaw = raw;
  }

  function updateScore() {
    try {
      updateTotalCharges();
      const raw = readFormValues();
      const result = scorer.scoreRaw(raw, model);
      renderResult(raw, result);
    } catch (error) {
      const probability = $("#probabilityValue");
      if (probability) {
        probability.textContent = "—";
      }
      showError(error.message || "The score could not be calculated.");
    }
  }

  /* Threshold lab ------------------------------------------------------- */
  const SVG_NS = "http://www.w3.org/2000/svg";
  const CHART = {
    width: 680,
    height: 270,
    left: 44,
    right: 14,
    top: 18,
    bottom: 32,
  };

  function svgElement(name, attributes) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes || {}).forEach(([key, value]) => {
      element.setAttribute(key, String(value));
    });
    return element;
  }

  function chartX(threshold) {
    const width = CHART.width - CHART.left - CHART.right;
    return CHART.left + threshold * width;
  }

  function chartY(value) {
    const height = CHART.height - CHART.top - CHART.bottom;
    return CHART.top + (1 - scorer.clamp(value, 0, 1)) * height;
  }

  function buildCurve() {
    const svg = $("#tradeoffChart");
    if (!svg) {
      return null;
    }
    const points = scorer.tradeoffCurve(model, 0.1, 0.9, 161);
    const title = $("#curveTitle");
    const description = $("#curveDesc");
    svg.replaceChildren();
    if (title) {
      svg.appendChild(title);
    }
    if (description) {
      svg.appendChild(description);
    }

    const grid = svgElement("g", { "aria-hidden": "true" });
    [0, 0.25, 0.5, 0.75, 1].forEach((value) => {
      const y = chartY(value);
      grid.appendChild(
        svgElement("line", {
          class: "grid-line",
          x1: CHART.left,
          x2: CHART.width - CHART.right,
          y1: y,
          y2: y,
        })
      );
      const label = svgElement("text", {
        x: CHART.left - 9,
        y: y + 3.5,
        "text-anchor": "end",
      });
      label.textContent = `${Math.round(value * 100)}%`;
      grid.appendChild(label);
    });

    [0.2, 0.4, 0.6, 0.8].forEach((value) => {
      const x = chartX(value);
      grid.appendChild(
        svgElement("line", {
          class: "grid-line",
          x1: x,
          x2: x,
          y1: CHART.top,
          y2: CHART.height - CHART.bottom,
        })
      );
    });

    const xAxis = svgElement("line", {
      class: "axis-line",
      x1: CHART.left,
      x2: CHART.width - CHART.right,
      y1: CHART.height - CHART.bottom,
      y2: CHART.height - CHART.bottom,
    });
    grid.appendChild(xAxis);
    [0.1, 0.3, 0.5, 0.7, 0.9].forEach((value) => {
      const label = svgElement("text", {
        x: chartX(value),
        y: CHART.height - 10,
        "text-anchor": "middle",
      });
      label.textContent = value.toFixed(1);
      grid.appendChild(label);
    });
    svg.appendChild(grid);

    const precisionPoints = points
      .map((point) => `${chartX(point.threshold).toFixed(2)},${chartY(point.precision).toFixed(2)}`)
      .join(" ");
    const recallPoints = points
      .map((point) => `${chartX(point.threshold).toFixed(2)},${chartY(point.recall).toFixed(2)}`)
      .join(" ");
    svg.appendChild(
      svgElement("polyline", {
        class: "precision-line",
        points: precisionPoints,
      })
    );
    svg.appendChild(
      svgElement("polyline", {
        class: "recall-line",
        points: recallPoints,
      })
    );

    const line = svgElement("line", {
      class: "threshold-line",
      id: "chartThresholdLine",
      x1: chartX(state.threshold),
      x2: chartX(state.threshold),
      y1: CHART.top,
      y2: CHART.height - CHART.bottom,
    });
    const precisionDot = svgElement("circle", {
      class: "threshold-dot",
      id: "chartPrecisionDot",
      r: 4,
    });
    const recallDot = svgElement("circle", {
      class: "threshold-dot",
      id: "chartRecallDot",
      r: 4,
    });
    const thresholdLabel = svgElement("text", {
      class: "current-threshold-label",
      id: "chartThresholdLabel",
      "text-anchor": "middle",
    });
    svg.append(line, precisionDot, recallDot, thresholdLabel);
    return points;
  }

  function updateChart(metrics) {
    const line = $("#chartThresholdLine");
    const precisionDot = $("#chartPrecisionDot");
    const recallDot = $("#chartRecallDot");
    const label = $("#chartThresholdLabel");
    if (!line || !precisionDot || !recallDot || !label) {
      return;
    }
    const x = chartX(state.threshold);
    line.setAttribute("x1", x);
    line.setAttribute("x2", x);
    precisionDot.setAttribute("cx", x);
    precisionDot.setAttribute("cy", chartY(metrics.precision));
    recallDot.setAttribute("cx", x);
    recallDot.setAttribute("cy", chartY(metrics.recall));
    label.setAttribute("x", x);
    label.setAttribute("y", CHART.top - 5);
    label.textContent = state.threshold.toFixed(3);
  }

  function renderThreshold(threshold) {
    state.threshold = threshold;
    const metrics = scorer.thresholdMetrics(model, threshold);
    const value = $("#thresholdValue");
    if (value) {
      value.textContent = threshold.toFixed(3);
    }
    const chip = $(".threshold-default");
    if (chip) {
      chip.textContent =
        Math.abs(threshold - model.metadata.threshold) < 0.0005
          ? "selected policy"
          : "custom policy";
    }
    const assign = (selector, content) => {
      const element = $(selector);
      if (element) {
        element.textContent = content;
      }
    };
    assign("#labContacts", metrics.contacts.toLocaleString("en-US"));
    assign("#labAlertRate", `${scorer.formatPct(metrics.alertRate)} alert rate`);
    assign("#labPrecision", scorer.formatPct(metrics.precision));
    assign(
      "#labPrecisionDetail",
      `${metrics.tp.toLocaleString("en-US")} of ${metrics.contacts.toLocaleString(
        "en-US"
      )} contacted`
    );
    assign("#labRecall", scorer.formatPct(metrics.recall));
    assign(
      "#labRecallDetail",
      `${metrics.tp.toLocaleString("en-US")} of ${model.holdout.churners.toLocaleString(
        "en-US"
      )} churners`
    );
    assign("#labF1", metrics.f1.toFixed(3));
    assign(
      "#labMissed",
      `${metrics.churnersMissed.toLocaleString("en-US")} churners missed`
    );
    assign(
      "#labRevenue",
      scorer.formatMoneyCompact(metrics.expectedAnnualRevenueAtRisk)
    );
    assign("#labTN", metrics.tn.toLocaleString("en-US"));
    assign("#labFP", metrics.fp.toLocaleString("en-US"));
    assign("#labFN", metrics.fn.toLocaleString("en-US"));
    assign("#labTP", metrics.tp.toLocaleString("en-US"));

    const hint = $("#thresholdHint");
    if (hint) {
      if (Math.abs(threshold - model.metadata.threshold) < 0.0005) {
        hint.replaceChildren();
        const strong = document.createElement("strong");
        strong.textContent =
          "The selected policy maximizes F1 on out-of-fold training predictions";
        hint.append(
          strong,
          document.createTextNode(
            ", then is evaluated once on the holdout set."
          )
        );
      } else if (threshold < model.metadata.threshold) {
        hint.textContent =
          "Lowering the threshold contacts more accounts and catches more churners, usually at the cost of precision. Replayed on the holdout set.";
      } else {
        hint.textContent =
          "Raising the threshold protects outreach capacity but misses more churners. Replayed on the holdout set.";
      }
    }

    const footnote = $("#thresholdFootnote");
    if (footnote) {
      footnote.replaceChildren();
      const strong = document.createElement("strong");
      strong.textContent = threshold.toFixed(3);
      footnote.append(
        document.createTextNode("At "),
        strong,
        document.createTextNode(
          `, ${metrics.tp.toLocaleString(
            "en-US"
          )} of ${model.holdout.churners.toLocaleString(
            "en-US"
          )} churners are contacted while ${metrics.fp.toLocaleString(
            "en-US"
          )} retained customers are also contacted. That is a deliberate retention trade-off, not an accuracy claim.`
        )
      );
    }
    updateChart(metrics);
  }

  /* Batch --------------------------------------------------------------- */
  function downloadFile(filename, text, mime) {
    const blob = new Blob(["\ufeff", text], {
      type: mime || "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function resetBatchView() {
    state.batchRows = null;
    state.batchFileName = "";
    const summary = $("#batchSummary");
    const table = $("#tableWrap");
    const empty = $("#batchEmpty");
    const foot = $("#tableFoot");
    const download = $("#downloadBatch");
    if (summary) summary.classList.remove("is-visible");
    if (table) table.classList.remove("is-visible");
    if (empty) empty.hidden = false;
    if (foot) foot.classList.remove("is-hidden");
    if (download) download.disabled = true;
  }

  function showBatchError(message) {
    resetBatchView();
    const status = $("#batchStatus");
    if (status) {
      status.textContent = message;
      status.className = "batch-status is-error";
    }
  }

  function batchStatus(message, type) {
    const status = $("#batchStatus");
    if (status) {
      status.textContent = message;
      status.className = "batch-status";
      if (type) {
        status.classList.add(type);
      }
    }
  }

  function renderBatch(result, fileName) {
    state.batchRows = result.rows;
    state.batchFileName = fileName;
    const flagged = result.rows.filter((row) => row.score.flagged).length;
    const critical = result.rows.filter(
      (row) => row.score.riskTier === "Critical"
    ).length;
    const revenue = result.rows.reduce(
      (total, row) => total + row.score.expectedAnnualRevenueAtRisk,
      0
    );
    const assign = (selector, content) => {
      const element = $(selector);
      if (element) {
        element.textContent = content;
      }
    };
    assign("#batchRows", result.rows.length.toLocaleString("en-US"));
    assign("#batchFlagged", flagged.toLocaleString("en-US"));
    assign("#batchCritical", critical.toLocaleString("en-US"));
    assign("#batchRevenue", scorer.formatMoneyCompact(revenue));

    const body = $("#batchBody");
    if (body) {
      body.replaceChildren();
      result.rows.slice(0, 25).forEach((row) => {
        const tr = document.createElement("tr");
        const values = [
          row.customerID,
          `${(row.score.probability * 100).toFixed(1)}%`,
          row.score.flagged ? "Flagged" : "Below",
          row.score.riskTier,
          `${Math.round(row.features.tenure)} mo`,
          row.features.Contract,
          row.features.InternetService,
          scorer.formatMoney(row.features.MonthlyCharges),
          scorer.formatMoney(row.score.expectedAnnualRevenueAtRisk),
        ];
        values.forEach((value, index) => {
          const cell = document.createElement("td");
          if (index === 0) {
            cell.className = "customer-id";
          } else if (index === 1) {
            cell.className = "probability-cell";
          }
          if (index === 2) {
            const badge = document.createElement("span");
            badge.className = `table-badge${row.score.flagged ? " flagged" : ""}`;
            badge.textContent = value;
            cell.appendChild(badge);
          } else if (index === 3) {
            const badge = document.createElement("span");
            badge.className = "table-badge";
            badge.style.setProperty("--badge", row.score.riskColor);
            badge.textContent = value;
            cell.appendChild(badge);
          } else {
            cell.textContent = value;
          }
          tr.appendChild(cell);
        });
        body.appendChild(tr);
      });
    }

    const caption = $("#tableCaption");
    if (caption) {
      caption.textContent = `Showing ${Math.min(
        25,
        result.rows.length
      ).toLocaleString("en-US")} of ${result.rows.length.toLocaleString(
        "en-US"
      )} accounts, ranked by churn probability.`;
    }
    const summary = $("#batchSummary");
    const table = $("#tableWrap");
    const empty = $("#batchEmpty");
    const foot = $("#tableFoot");
    const download = $("#downloadBatch");
    if (summary) summary.classList.add("is-visible");
    if (table) table.classList.add("is-visible");
    if (empty) empty.hidden = true;
    if (foot) foot.classList.remove("is-hidden");
    if (download) download.disabled = false;

    const unknownNote =
      result.unknownCategories > 0
        ? ` ${result.unknownCategories.toLocaleString(
            "en-US"
          )} unknown category value(s) were ignored, matching the pipeline's handling.`
        : "";
    batchStatus(
      `Scored ${result.rows.length.toLocaleString("en-US")} customers from ${fileName}.${unknownNote}`,
      "is-success"
    );
  }

  function ingestCsv(text, fileName) {
    try {
      const parsed = scorer.parseCsv(text);
      const result = scorer.cleanBatchRecords(parsed.records, model);
      renderBatch(result, fileName);
    } catch (error) {
      state.batchRows = null;
      const download = $("#downloadBatch");
      if (download) download.disabled = true;
      showBatchError(error.message || "The CSV could not be scored.");
    }
  }

  function handleFile(file) {
    if (!file) {
      return;
    }
    batchStatus(`Reading ${file.name}…`);
    file
      .text()
      .then((text) => ingestCsv(text, file.name))
      .catch(() => showBatchError("The file could not be read."));
  }

  function setupBatch() {
    const fileInput = $("#fileInput");
    const dropzone = $("#dropzone");
    if (fileInput) {
      fileInput.addEventListener("change", () => {
        handleFile(fileInput.files && fileInput.files[0]);
      });
    }
    if (dropzone) {
      ["dragenter", "dragover"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
          event.preventDefault();
          dropzone.classList.add("is-dragging");
        });
      });
      ["dragleave", "drop"].forEach((eventName) => {
        dropzone.addEventListener(eventName, (event) => {
          event.preventDefault();
          dropzone.classList.remove("is-dragging");
        });
      });
      dropzone.addEventListener("drop", (event) => {
        const file = event.dataTransfer.files && event.dataTransfer.files[0];
        if (file && fileInput) {
          handleFile(file);
        }
      });
    }

    const loadSample = $("#loadSample");
    if (loadSample) {
      loadSample.addEventListener("click", () => {
        ingestCsv(scorer.sampleCsv(model), "sample_customers.csv");
      });
    }

    const downloadTemplate = $("#downloadTemplate");
    if (downloadTemplate) {
      downloadTemplate.addEventListener("click", () => {
        try {
          const csv = scorer.templateCsv(model, readFormValues());
          downloadFile("churn_batch_template.csv", csv);
        } catch (error) {
          showBatchError(error.message || "The template could not be created.");
        }
      });
    }

    const downloadBatch = $("#downloadBatch");
    if (downloadBatch) {
      downloadBatch.addEventListener("click", () => {
        if (!state.batchRows) {
          return;
        }
        const date = new Date().toISOString().slice(0, 10);
        downloadFile(
          `churn_scored_${date}.csv`,
          scorer.batchCsv(state.batchRows)
        );
      });
    }
  }

  function setupFormEvents() {
    const form = $("#profileForm");
    if (form) {
      form.addEventListener("input", (event) => {
        if (event.target && event.target.id === "autoTotal") {
          updateTotalCharges();
        } else if (
          event.target &&
          (event.target.id === "field-tenure" ||
            event.target.id === "field-MonthlyCharges")
        ) {
          updateTotalCharges();
        }
        updateScore();
      });
      form.addEventListener("change", updateScore);
      form.addEventListener("submit", (event) => event.preventDefault());
    }
    ["#exampleHigh", "#exampleLoyal"].forEach((selector) => {
      const button = $(selector);
      if (button) {
        button.addEventListener("click", () =>
          loadExample(selector === "#exampleHigh" ? "new-fiber" : "loyal")
        );
      }
    });
    const reset = $("#resetForm");
    if (reset) {
      reset.addEventListener("click", () => loadExample("new-fiber"));
    }
  }

  function setupThreshold() {
    const slider = $("#thresholdSlider");
    if (!slider) {
      return;
    }
    slider.value = model.metadata.threshold.toFixed(3);
    state.threshold = model.metadata.threshold;
    buildCurve();
    renderThreshold(model.metadata.threshold);
    slider.addEventListener("input", () => {
      renderThreshold(Number(slider.value));
    });
  }

  function init() {
    if (!model || !scorer) {
      showError(
        "The exported model data could not be loaded. Open the page through the deployed site or a local web server."
      );
      const status = $("#batchStatus");
      if (status) {
        status.textContent =
          "The scorer data file did not load; batch scoring is unavailable.";
        status.className = "batch-status is-error";
      }
      return;
    }
    renderForm();
    setupFormEvents();
    setupBatch();
    hydrateMetadata();
    setupThreshold();
    const form = $("#profileForm");
    if (form) {
      form.reset();
    }
    loadExample("new-fiber");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
