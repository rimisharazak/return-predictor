// Category -> sub-category options, matching the categories seen in training data
const CATEGORY_MAP = {
  "Fashion": ["Women", "Accessories", "Footwear", "Men", "Kids"],
  "Electronics": ["Mobiles", "Audio", "Laptops"],
  "Home": ["Furniture", "Kitchen", "Decor"],
  "Beauty": ["Haircare", "Skincare", "Makeup"],
  "Sports": ["Team Sports", "Fitness", "Outdoor"],
  "Books": ["Non-Fiction", "Fiction", "Academic"],
  "Toys": ["Outdoor Play", "Educational", "Action Figures"],
};

const BRANDS = Array.from({ length: 30 }, (_, i) => `Brand_${i + 1}`);

function populateSelect(select, options) {
  select.innerHTML = "";
  options.forEach((opt) => {
    const el = document.createElement("option");
    el.value = opt;
    el.textContent = opt;
    select.appendChild(el);
  });
}

function initCategoryDropdowns() {
  const categorySelect = document.getElementById("product_category");
  const subCategorySelect = document.getElementById("sub_category");

  populateSelect(categorySelect, Object.keys(CATEGORY_MAP));
  populateSelect(subCategorySelect, CATEGORY_MAP[categorySelect.value]);
  categorySelect.value = "Electronics";
  populateSelect(subCategorySelect, CATEGORY_MAP["Electronics"]);
  subCategorySelect.value = "Mobiles";

  categorySelect.addEventListener("change", () => {
    populateSelect(subCategorySelect, CATEGORY_MAP[categorySelect.value]);
  });
}

function initBrandDropdown() {
  const brandSelect = document.getElementById("brand");
  populateSelect(brandSelect, BRANDS);
  brandSelect.value = "Brand_1";
}

function collectPayload(form) {
  const data = new FormData(form);
  const payload = {};
  for (const [key, value] of data.entries()) {
    payload[key] = value;
  }
  // checkboxes not present in FormData when unchecked -> explicitly set booleans
  ["fragile_item", "warranty_available", "delayed_delivery", "wishlist_before_purchase"].forEach((id) => {
    payload[id] = document.getElementById(id).checked;
  });
  return payload;
}

function setGauge(percent, riskLevel) {
  const arc = document.getElementById("gauge-arc");
  const circumference = 251.2; // matches path length used in CSS/SVG
  const offset = circumference - (circumference * Math.min(percent, 100)) / 100;
  arc.style.strokeDashoffset = offset;

  let color = "var(--accent-red)";
  let badgeClass = "";
  if (riskLevel === "Low Risk") {
    color = "var(--accent-green)";
    badgeClass = "low";
  } else if (riskLevel === "Medium Risk") {
    color = "var(--accent-orange)";
    badgeClass = "medium";
  }
  arc.style.stroke = color;

  document.getElementById("gauge-percent").textContent = `${percent}%`;
  const badge = document.getElementById("risk-badge");
  badge.textContent = riskLevel;
  badge.className = `risk-badge ${badgeClass}`;
}

async function handleSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const payload = collectPayload(form);

  const submitBtn = form.querySelector(".btn-predict");
  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();

    if (!res.ok) {
      alert(result.error || "Prediction failed.");
      return;
    }

    document.getElementById("result-empty").hidden = true;
    document.getElementById("result-content").hidden = false;
    setGauge(result.probability, result.risk_level);
  } catch (err) {
    alert("Could not reach the prediction server: " + err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "⚡ Predict";
  }
}

function handleReset() {
  document.getElementById("result-empty").hidden = false;
  document.getElementById("result-content").hidden = true;
  setTimeout(() => {
    initCategoryDropdowns();
    initBrandDropdown();
  }, 0);
}

document.addEventListener("DOMContentLoaded", () => {
  initCategoryDropdowns();
  initBrandDropdown();
  document.getElementById("predict-form").addEventListener("submit", handleSubmit);
  document.getElementById("reset-btn").addEventListener("click", handleReset);
});
