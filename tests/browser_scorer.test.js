/* Node parity checks for the static browser scorer.
 *
 * Run with: node tests/browser_scorer.test.js
 */
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");
const assert = require("assert");

const root = path.resolve(__dirname, "..");
const scorer = require(path.join(root, "docs", "scorer.js"));
const sandbox = { window: {} };
vm.createContext(sandbox);
vm.runInContext(
  fs.readFileSync(path.join(root, "docs", "scorer_data.js"), "utf8"),
  sandbox
);
const artifact = sandbox.window.CHURN_MODEL;

assert.ok(artifact, "scorer data must be present");
assert.strictEqual(artifact.numeric.length + artifact.categorical.length, 22);
assert.strictEqual(artifact.numeric.length, 6);
assert.strictEqual(artifact.categorical.length, 16);

for (const validation of artifact.validation) {
  const result = scorer.scoreFeatures(validation.features, artifact);
  assert.ok(
    Math.abs(result.rawProbability - validation.expectedProbability) < 1e-12,
    `${validation.id}: expected ${validation.expectedProbability}, got ${result.rawProbability}`
  );
  assert.ok(
    result.rawProbability >= 0 && result.rawProbability <= 1,
    `${validation.id}: probability must be bounded`
  );
}

for (const example of artifact.examples) {
  const result = scorer.scoreRaw(example.raw, artifact);
  const validation = artifact.validation.find((item) => item.id === example.id);
  assert.ok(validation, `${example.id}: validation case is missing`);
  assert.ok(
    Math.abs(result.rawProbability - validation.expectedProbability) < 1e-12,
    `${example.id}: raw example drifted from the exported validation case`
  );
  assert.strictEqual(result.features.ServiceCount >= 0, true);
  assert.strictEqual(
    result.features.IsNewCustomer,
    result.features.tenure <= 6 ? 1 : 0
  );
}

const selected = scorer.thresholdMetrics(
  artifact,
  artifact.metadata.threshold
);
assert.strictEqual(selected.contacts, 492);
assert.strictEqual(selected.tp, 270);
assert.strictEqual(selected.fp, 222);
assert.strictEqual(selected.fn, 104);
assert.strictEqual(selected.tn, 813);
assert.ok(Math.abs(selected.precision - 0.5488) < 0.0001);
assert.ok(Math.abs(selected.recall - 0.7219) < 0.0001);

const sample = scorer.parseCsv(scorer.sampleCsv(artifact));
const batch = scorer.cleanBatchRecords(sample.records, artifact);
assert.strictEqual(batch.rows.length, artifact.examples.length);
assert.strictEqual(batch.rows[0].customerID, "DEMO-001");
assert.strictEqual(batch.rows[0].features.Contract, "Month-to-month");
assert.strictEqual(
  scorer.batchHeaders().length,
  28,
  "campaign download should preserve the 28-column contract"
);

const blankTotal = scorer.parseCsv(
  "gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges\n" +
    "Female,No,No,No,2,Yes,Yes,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,80,\n"
);
const derived = scorer.cleanBatchRecords(blankTotal.records, artifact);
assert.strictEqual(derived.rows[0].features.TotalCharges, 160);
assert.strictEqual(derived.rows[0].features.AvgMonthlySpend, 80);

let duplicateError = null;
try {
  scorer.cleanBatchRecords(
    [
      { ...artifact.examples[0].raw, customerID: "same" },
      { ...artifact.examples[1].raw, customerID: "same" },
    ],
    artifact
  );
} catch (error) {
  duplicateError = error;
}
assert.ok(duplicateError, "duplicate IDs should be rejected");

console.log("Browser scorer parity checks passed.");
