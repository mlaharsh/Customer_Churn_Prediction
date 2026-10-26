/* Churn Signal browser scorer.
 *
 * This file mirrors the fitted scikit-learn pipeline exported in
 * scorer_data.js. It has no network calls and no third-party dependencies.
 * The CommonJS export is intentional: tests/browser_scorer.test.js exercises
 * exactly the same functions the page uses.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.ChurnScorer = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const RAW_FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
  ];

  const SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
  ];

  const MAX_BATCH_ROWS = 50000;

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function roundTo(value, digits) {
    const factor = 10 ** digits;
    return Math.round((value + Number.EPSILON) * factor) / factor;
  }

  function sigmoid(logit) {
    if (logit >= 0) {
      const z = Math.exp(-logit);
      return 1 / (1 + z);
    }
    const z = Math.exp(logit);
    return z / (1 + z);
  }

  function normalizeSeniorCitizen(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric) && (numeric === 0 || numeric === 1)) {
      return numeric === 1 ? "Yes" : "No";
    }
    const text = String(value == null ? "" : value)
      .trim()
      .toLowerCase();
    if (text === "yes" || text === "true") {
      return "Yes";
    }
    return "No";
  }

  function numericValue(value) {
    if (value === null || value === undefined || String(value).trim() === "") {
      return NaN;
    }
    const number = Number(value);
    return Number.isFinite(number) ? number : NaN;
  }

  function featureMap(artifact) {
    return {
      numeric: new Map(artifact.numeric.map((feature) => [feature.name, feature])),
      categorical: new Map(
        artifact.categorical.map((feature) => [feature.name, feature])
      ),
    };
  }

  function deriveFeatures(raw, artifact) {
    const tenure = numericValue(raw.tenure);
    const monthlyCharges = numericValue(raw.MonthlyCharges);
    if (!Number.isFinite(tenure)) {
      throw new Error("Tenure must be a number.");
    }
    if (tenure < 0) {
      throw new Error("Tenure cannot be negative.");
    }
    if (!Number.isFinite(monthlyCharges)) {
      throw new Error("Monthly charges must be a number.");
    }
    if (monthlyCharges < 0) {
      throw new Error("Monthly charges cannot be negative.");
    }

    let totalCharges = numericValue(raw.TotalCharges);
    if (!Number.isFinite(totalCharges)) {
      totalCharges = monthlyCharges * tenure;
    }
    const averageMonthlySpend =
      tenure > 0 ? totalCharges / tenure : monthlyCharges;
    const serviceCount = SERVICE_COLUMNS.reduce(
      (count, name) => count + (raw[name] === "Yes" ? 1 : 0),
      0
    );

    const features = {
      tenure: tenure,
      MonthlyCharges: monthlyCharges,
      TotalCharges: totalCharges,
      AvgMonthlySpend: averageMonthlySpend,
      ServiceCount: serviceCount,
      IsNewCustomer: tenure <= 6 ? 1 : 0,
    };

    for (const feature of artifact.categorical) {
      let value = raw[feature.name];
      if (value === null || value === undefined) {
        value = "";
      }
      features[feature.name] =
        feature.name === "SeniorCitizen"
          ? normalizeSeniorCitizen(value)
          : String(value);
    }
    return features;
  }

  function riskBand(probability, artifact) {
    const bands = artifact.metadata.riskBands;
    for (const band of bands) {
      if (probability >= band.min && probability < band.max) {
        return band;
      }
    }
    return bands[bands.length - 1];
  }

  function scoreFeatures(features, artifact) {
    let logit = artifact.intercept;

    for (const feature of artifact.numeric) {
      let value = numericValue(features[feature.name]);
      if (!Number.isFinite(value)) {
        value = feature.median;
      }
      logit +=
        ((value - feature.mean) / feature.scale) * feature.coefficient;
    }

    for (const feature of artifact.categorical) {
      const value = String(features[feature.name]);
      if (Object.prototype.hasOwnProperty.call(feature.coefficients, value)) {
        logit += feature.coefficients[value];
      }
      // Unknown levels are ignored, matching OneHotEncoder(handle_unknown="ignore").
    }

    const rawProbability = sigmoid(logit);
    const probability = roundTo(rawProbability, 4);
    const band = riskBand(rawProbability, artifact);
    const flagged = rawProbability >= artifact.metadata.threshold;
    const monthlyRevenueAtRisk = roundTo(
      features.MonthlyCharges * probability,
      2
    );
    const annualRevenueAtRisk = roundTo(monthlyRevenueAtRisk * 12, 2);

    return {
      logit: logit,
      rawProbability: rawProbability,
      probability: probability,
      riskTier: band.name,
      riskColor: band.color,
      flagged: flagged,
      expectedMonthlyRevenueAtRisk: monthlyRevenueAtRisk,
      expectedAnnualRevenueAtRisk: annualRevenueAtRisk,
    };
  }

  function scoreRaw(raw, artifact) {
    const features = deriveFeatures(raw, artifact);
    return {
      features: features,
      ...scoreFeatures(features, artifact),
    };
  }

  function retentionActions(features, probability, artifact) {
    const threshold = artifact.metadata.threshold;
    const actions = [];

    if (features.Contract === "Month-to-month") {
      actions.push(
        "Offer a one-year contract with a loyalty credit; contract length is the strongest segment-level churn signal."
      );
    }
    if (Number(features.IsNewCustomer) === 1) {
      actions.push(
        "Trigger a 30-day onboarding check-in and confirm the customer has activated their core services."
      );
    }
    if (
      features.TechSupport === "No" ||
      features.OnlineSecurity === "No"
    ) {
      actions.push(
        "Bundle a no-cost tech-support or online-security trial; missing support is associated with higher churn."
      );
    }
    if (features.InternetService === "Fiber optic") {
      actions.push(
        "Run a fiber service-quality and price-to-value review; fiber customers show elevated churn in this dataset."
      );
    }
    if (features.PaymentMethod === "Electronic check") {
      actions.push(
        "Incentivize automatic bank or card payment to reduce friction and improve renewal stickiness."
      );
    }
    if (Number(features.MonthlyCharges) >= 80) {
      actions.push(
        "Review the plan against usage and offer a right-sized bundle rather than a blanket discount."
      );
    }
    if (probability >= threshold) {
      actions.unshift(
        "Assign a retention specialist within 48 hours and document the save offer outcome."
      );
    }
    if (actions.length === 0) {
      actions.push(
        "Maintain standard engagement; monitor the next billing cycle for changes in tenure or service mix."
      );
    }
    return actions.slice(0, 4);
  }

  function displayFeatureValue(featureName, value) {
    if (featureName === "IsNewCustomer") {
      return Number(value) === 1 ? "Yes" : "No";
    }
    if (featureName === "tenure") {
      return `${Math.round(Number(value))} months`;
    }
    if (
      featureName === "MonthlyCharges" ||
      featureName === "TotalCharges" ||
      featureName === "AvgMonthlySpend"
    ) {
      return formatMoney(Number(value));
    }
    if (featureName === "ServiceCount") {
      return `${Math.round(Number(value))} active`;
    }
    return String(value);
  }

  function whatIf(features, artifact, topN) {
    const limit = topN || 5;
    const current = scoreFeatures(features, artifact).rawProbability;
    const records = [];

    for (const feature of [...artifact.numeric, ...artifact.categorical]) {
      const baseline = artifact.baselines[feature.name];
      if (baseline === undefined) {
        continue;
      }
      const candidate = { ...features, [feature.name]: baseline };
      const alternative = scoreFeatures(candidate, artifact).rawProbability;
      records.push({
        feature: feature.name,
        label: feature.label,
        current: displayFeatureValue(feature.name, features[feature.name]),
        baseline: displayFeatureValue(feature.name, baseline),
        impactPp: roundTo((current - alternative) * 100, 1),
      });
    }

    return records
      .sort((left, right) => Math.abs(right.impactPp) - Math.abs(left.impactPp))
      .slice(0, limit);
  }

  function thresholdMetrics(artifact, threshold) {
    const holdout = artifact.holdout;
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;
    let revenueAtRisk = 0;

    for (let index = 0; index < holdout.probabilities.length; index += 1) {
      const probability = holdout.probabilities[index];
      const actual = holdout.labels.charAt(index) === "1";
      const contacted = probability >= threshold;
      if (contacted) {
        revenueAtRisk += holdout.monthlyCharges[index] * probability * 12;
        if (actual) {
          tp += 1;
        } else {
          fp += 1;
        }
      } else if (actual) {
        fn += 1;
      } else {
        tn += 1;
      }
    }

    const contacts = tp + fp;
    const precision = contacts > 0 ? tp / contacts : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;
    const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
    const accuracy = (tp + tn) / Math.max(1, tp + tn + fp + fn);

    return {
      threshold: threshold,
      contacts: contacts,
      alertRate: contacts / Math.max(1, holdout.n),
      precision: precision,
      recall: recall,
      f1: f1,
      specificity: specificity,
      accuracy: accuracy,
      tp: tp,
      fp: fp,
      fn: fn,
      tn: tn,
      churnersMissed: fn,
      nonChurnersContacted: fp,
      expectedAnnualRevenueAtRisk: roundTo(revenueAtRisk, 2),
    };
  }

  function tradeoffCurve(artifact, minimum, maximum, steps) {
    const count = steps || 81;
    const points = [];
    for (let index = 0; index < count; index += 1) {
      const threshold =
        minimum + ((maximum - minimum) * index) / Math.max(1, count - 1);
      points.push(thresholdMetrics(artifact, threshold));
    }
    return points;
  }

  function parseCsv(text) {
    if (typeof text !== "string" || text.trim() === "") {
      throw new Error("The CSV file is empty.");
    }
    const source = text.replace(/^\uFEFF/, "");
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;

    for (let index = 0; index < source.length; index += 1) {
      const character = source[index];
      if (inQuotes) {
        if (character === '"') {
          if (source[index + 1] === '"') {
            field += '"';
            index += 1;
          } else {
            inQuotes = false;
          }
        } else {
          field += character;
        }
        continue;
      }

      if (character === '"') {
        inQuotes = true;
      } else if (character === ",") {
        row.push(field);
        field = "";
      } else if (character === "\n") {
        row.push(field);
        rows.push(row);
        row = [];
        field = "";
      } else if (character !== "\r") {
        field += character;
      }
    }

    if (inQuotes) {
      throw new Error("The CSV contains an unterminated quoted field.");
    }
    if (field.length > 0 || row.length > 0) {
      row.push(field);
      rows.push(row);
    }

    while (rows.length > 0 && rows[rows.length - 1].every((value) => value === "")) {
      rows.pop();
    }
    if (rows.length < 2) {
      throw new Error("The CSV needs a header row and at least one data row.");
    }

    const headers = rows[0].map((header) => header.trim());
    if (headers.some((header) => header === "")) {
      throw new Error("The CSV contains an empty column name.");
    }
    if (new Set(headers).size !== headers.length) {
      throw new Error("The CSV contains duplicate column names.");
    }

    const records = [];
    for (let rowIndex = 1; rowIndex < rows.length; rowIndex += 1) {
      const values = rows[rowIndex];
      if (values.every((value) => value === "")) {
        continue;
      }
      const record = {};
      headers.forEach((header, columnIndex) => {
        record[header] = values[columnIndex] === undefined ? "" : values[columnIndex];
      });
      records.push(record);
    }

    if (records.length === 0) {
      throw new Error("The CSV does not contain any data rows.");
    }
    if (records.length > MAX_BATCH_ROWS) {
      throw new Error(
        `The CSV contains ${records.length.toLocaleString()} rows; the browser limit is ${MAX_BATCH_ROWS.toLocaleString()}.`
      );
    }
    return { headers: headers, records: records };
  }

  function countUnknownCategories(features, artifact) {
    let unknown = 0;
    for (const feature of artifact.categorical) {
      if (
        !Object.prototype.hasOwnProperty.call(
          feature.coefficients,
          String(features[feature.name])
        )
      ) {
        unknown += 1;
      }
    }
    return unknown;
  }

  function cleanBatchRecords(records, artifact) {
    if (!Array.isArray(records) || records.length === 0) {
      throw new Error("No customer rows were found.");
    }
    if (records.length > MAX_BATCH_ROWS) {
      throw new Error(
        `The upload contains ${records.length.toLocaleString()} rows; the browser limit is ${MAX_BATCH_ROWS.toLocaleString()}.`
      );
    }

    const available = new Set(Object.keys(records[0]));
    const missing = RAW_FEATURE_COLUMNS.filter((column) => !available.has(column));
    if (missing.length > 0) {
      throw new Error(`Missing required columns: ${missing.join(", ")}`);
    }

    const seenIds = new Set();
    let unknownCategories = 0;
    const scored = [];

    records.forEach((record, index) => {
      const suppliedId = String(record.customerID == null ? "" : record.customerID).trim();
      const customerID = suppliedId || `row-${index + 1}`;
      if (seenIds.has(customerID)) {
        throw new Error(`Duplicate customerID: ${customerID}`);
      }
      seenIds.add(customerID);

      const raw = {};
      for (const column of RAW_FEATURE_COLUMNS) {
        raw[column] = record[column] === undefined ? "" : record[column];
      }
      const features = deriveFeatures(raw, artifact);
      const score = scoreFeatures(features, artifact);
      unknownCategories += countUnknownCategories(features, artifact);

      scored.push({
        customerID: customerID,
        raw: raw,
        features: features,
        score: score,
      });
    });

    scored.sort(
      (left, right) => right.score.rawProbability - left.score.rawProbability
    );
    return { rows: scored, unknownCategories: unknownCategories };
  }

  function batchOutputRows(results) {
    return results.map((result) => {
      const row = {
        customerID: result.customerID,
        churn_probability: result.score.probability,
        churn_prediction: result.score.flagged ? 1 : 0,
        risk_tier: result.score.riskTier,
        expected_monthly_revenue_at_risk:
          result.score.expectedMonthlyRevenueAtRisk,
        expected_annual_revenue_at_risk:
          result.score.expectedAnnualRevenueAtRisk,
      };
      for (const column of RAW_FEATURE_COLUMNS) {
        row[column] = result.raw[column];
      }
      row.AvgMonthlySpend = roundTo(result.features.AvgMonthlySpend, 4);
      row.ServiceCount = result.features.ServiceCount;
      row.IsNewCustomer = result.features.IsNewCustomer;
      return row;
    });
  }

  function batchHeaders() {
    return [
      "customerID",
      "churn_probability",
      "churn_prediction",
      "risk_tier",
      "expected_monthly_revenue_at_risk",
      "expected_annual_revenue_at_risk",
      ...RAW_FEATURE_COLUMNS,
      "AvgMonthlySpend",
      "ServiceCount",
      "IsNewCustomer",
    ];
  }

  function escapeCsvValue(value) {
    if (value === null || value === undefined) {
      return "";
    }
    const text = String(value);
    if (/[",\r\n]/.test(text)) {
      return `"${text.replace(/"/g, '""')}"`;
    }
    return text;
  }

  function toCsv(headers, rows) {
    const lines = [headers.map(escapeCsvValue).join(",")];
    for (const row of rows) {
      lines.push(
        headers.map((header) => escapeCsvValue(row[header])).join(",")
      );
    }
    return `${lines.join("\r\n")}\r\n`;
  }

  function batchCsv(results) {
    return toCsv(batchHeaders(), batchOutputRows(results));
  }

  function templateCsv(artifact, raw) {
    const headers = ["customerID", ...RAW_FEATURE_COLUMNS];
    const row = { customerID: "CUST-1001" };
    for (const column of RAW_FEATURE_COLUMNS) {
      row[column] = raw[column] == null ? "" : raw[column];
    }
    return toCsv(headers, [row]);
  }

  function sampleCsv(artifact) {
    const headers = ["customerID", ...RAW_FEATURE_COLUMNS];
    const rows = artifact.examples.map((example, index) => {
      const row = { customerID: `DEMO-${String(index + 1).padStart(3, "0")}` };
      for (const column of RAW_FEATURE_COLUMNS) {
        row[column] = example.raw[column] == null ? "" : example.raw[column];
      }
      return row;
    });
    return toCsv(headers, rows);
  }

  function formatMoney(value) {
    return Number(value).toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  function formatMoneyCompact(value) {
    const number = Number(value);
    if (Math.abs(number) >= 1000000) {
      return `$${(number / 1000000).toFixed(2)}M`;
    }
    if (Math.abs(number) >= 1000) {
      return `$${(number / 1000).toFixed(1)}k`;
    }
    return formatMoney(number);
  }

  function formatPct(value, digits) {
    return `${(Number(value) * 100).toFixed(digits == null ? 1 : digits)}%`;
  }

  function formatPp(value) {
    const number = Number(value);
    const sign = number > 0 ? "+" : "";
    return `${sign}${number.toFixed(1)} pp`;
  }

  return {
    MAX_BATCH_ROWS: MAX_BATCH_ROWS,
    RAW_FEATURE_COLUMNS: RAW_FEATURE_COLUMNS,
    SERVICE_COLUMNS: SERVICE_COLUMNS,
    clamp: clamp,
    roundTo: roundTo,
    sigmoid: sigmoid,
    normalizeSeniorCitizen: normalizeSeniorCitizen,
    deriveFeatures: deriveFeatures,
    scoreFeatures: scoreFeatures,
    scoreRaw: scoreRaw,
    riskBand: riskBand,
    retentionActions: retentionActions,
    whatIf: whatIf,
    thresholdMetrics: thresholdMetrics,
    tradeoffCurve: tradeoffCurve,
    parseCsv: parseCsv,
    cleanBatchRecords: cleanBatchRecords,
    batchOutputRows: batchOutputRows,
    batchHeaders: batchHeaders,
    batchCsv: batchCsv,
    templateCsv: templateCsv,
    sampleCsv: sampleCsv,
    toCsv: toCsv,
    formatMoney: formatMoney,
    formatMoneyCompact: formatMoneyCompact,
    formatPct: formatPct,
    formatPp: formatPp,
    featureMap: featureMap,
  };
});
