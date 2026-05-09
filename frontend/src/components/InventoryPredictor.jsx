import { useState } from "react";

const API_URL     = "https://inventory-predictor-api-webapp.onrender.com/api/predict";
const RETRAIN_URL = "https://inventory-predictor-api-webapp.onrender.com/api/retrain";

const PRODUCT_CATEGORIES = [
  "Electronics",
  "Fashion",
  "Groceries & Beverages",
  "Gym Products",
  "Stationery & School Products",
];

const PRODUCTS = [
  { id: "P001", name: "iPhone 15" },
  { id: "P002", name: "Smart Watch" },
  { id: "P003", name: "Bluetooth Speakers" },
  { id: "P004", name: "Men's Winter Jacket" },
  { id: "P005", name: "Summer Cotton T-Shirts" },
  { id: "P006", name: "Festive Sarees" },
  { id: "P007", name: "Treadmill" },
  { id: "P008", name: "Dumbbell Set" },
  { id: "P009", name: "Yoga Mats" },
  { id: "P010", name: "Soft Drink Cases" },
  { id: "P011", name: "Packaged Snacks" },
  { id: "P012", name: "Tea & Coffee Packs" },
  { id: "P013", name: "Notebooks (Bundle)" },
  { id: "P014", name: "Pens & Markers Set" },
  { id: "P015", name: "School Backpacks" },
];

const initialForm = {
  price_inr: "",
  current_product_stock: "",
  product_category: "",
  product_id: "",
  order_date: "",
};

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    padding: "24px",
    fontFamily: "'Segoe UI', sans-serif",
  },
  wrapper: { width: "100%", maxWidth: "680px" },
  badge: {
    display: "inline-block",
    background: "#e0e7ff",
    color: "#4f46e5",
    borderRadius: "999px",
    padding: "4px 14px",
    fontSize: "11px",
    fontWeight: "700",
    letterSpacing: "2px",
    textTransform: "uppercase",
    marginBottom: "12px",
  },
  title: { fontSize: "32px", fontWeight: "900", color: "#0f172a", margin: "0 0 8px 0" },
  subtitle: { fontSize: "13px", color: "#94a3b8", margin: "0 0 32px 0" },
  card: {
    background: "#fff",
    borderRadius: "20px",
    boxShadow: "0 20px 60px rgba(99,102,241,0.1)",
    border: "1px solid #f1f5f9",
    padding: "36px",
  },
  retrainRow: { display: "flex", justifyContent: "flex-end", marginBottom: "20px" },
  retrainBtn: {
    background: "#f1f5f9",
    border: "none",
    borderRadius: "10px",
    padding: "8px 16px",
    fontSize: "12px",
    fontWeight: "700",
    color: "#475569",
    cursor: "pointer",
  },
  successMsg: {
    background: "#f0fdf4",
    border: "1px solid #bbf7d0",
    borderRadius: "12px",
    padding: "12px",
    fontSize: "13px",
    color: "#16a34a",
    textAlign: "center",
    marginBottom: "16px",
  },
  grid2: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px", marginBottom: "16px" },
  grid3: { display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px", marginBottom: "16px" },
  field: { display: "flex", flexDirection: "column" },
  label: {
    fontSize: "11px",
    fontWeight: "700",
    color: "#94a3b8",
    textTransform: "uppercase",
    letterSpacing: "1px",
    marginBottom: "6px",
  },
  input: {
    border: "1px solid #e2e8f0",
    borderRadius: "10px",
    padding: "10px 12px",
    fontSize: "13px",
    color: "#1e293b",
    outline: "none",
    background: "#fff",
    width: "100%",
    boxSizing: "border-box",
  },
  submitBtn: {
    width: "100%",
    background: "linear-gradient(135deg, #6366f1, #4f46e5)",
    border: "none",
    borderRadius: "14px",
    padding: "14px",
    fontSize: "14px",
    fontWeight: "700",
    color: "#fff",
    cursor: "pointer",
    marginTop: "8px",
    boxShadow: "0 8px 24px rgba(99,102,241,0.3)",
  },
  resultBox: {
    marginTop: "24px",
    background: "#eef2ff",
    border: "1px solid #c7d2fe",
    borderRadius: "16px",
    padding: "24px",
    textAlign: "center",
  },
  resultLabel: {
    fontSize: "11px",
    fontWeight: "700",
    color: "#818cf8",
    textTransform: "uppercase",
    letterSpacing: "2px",
    marginBottom: "8px",
  },
  resultNumber: { fontSize: "72px", fontWeight: "900", color: "#4338ca", lineHeight: 1 },
  resultSub: { fontSize: "11px", color: "#94a3b8", marginTop: "8px" },
  errorBox: {
    marginTop: "24px",
    background: "#fef2f2",
    border: "1px solid #fecaca",
    borderRadius: "12px",
    padding: "14px",
    fontSize: "13px",
    color: "#dc2626",
  },
  footer: { marginTop: "16px", textAlign: "center", fontSize: "11px", color: "#cbd5e1" },
};

export default function InventoryPredictor() {
  const [form, setForm]             = useState(initialForm);
  const [result, setResult]         = useState(null);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(null);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setResult(null);
    setError(null);
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg(null);
    setError(null);
    try {
      const res  = await fetch(RETRAIN_URL, { method: "POST" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Retrain failed");
      setRetrainMsg(data.status);
    } catch (err) {
      setError(err.message);
    } finally {
      setRetraining(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setRetrainMsg(null);

    const body = {
      price_inr:             parseFloat(form.price_inr),
      current_product_stock: parseInt(form.current_product_stock, 10),
      product_category:      form.product_category,
      product_id:            form.product_id,
      order_date:            form.order_date,
    };

    try {
      const res  = await fetch(API_URL, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.wrapper}>
        {/* Header */}
        <div style={{ textAlign: "center" }}>
          <span style={styles.badge}>Stockify Inventory AI</span>
          <h1 style={styles.title}>Stock Demand Predictor</h1>
          <p style={styles.subtitle}>Powered by Hist Gradient Boosting · R² = 0.732</p>
        </div>

        {/* Card */}
        <div style={styles.card}>

          {/* Retrain Button */}
          <div style={styles.retrainRow}>
            <button onClick={handleRetrain} disabled={retraining} style={styles.retrainBtn}>
              {retraining ? "Retraining …" : "🔄 Retrain Model"}
            </button>
          </div>

          {retrainMsg && <div style={styles.successMsg}>✓ {retrainMsg}</div>}

          <form onSubmit={handleSubmit}>
            {/* Row 1 */}
            <div style={styles.grid2}>
              <div style={styles.field}>
                <label style={styles.label}>Price (INR)</label>
                <input style={styles.input} type="number" name="price_inr" required
                  placeholder="e.g. 89900" value={form.price_inr} onChange={handleChange} />
              </div>
              <div style={styles.field}>
                <label style={styles.label}>Current Stock</label>
                <input style={styles.input} type="number" name="current_product_stock" required
                  placeholder="e.g. 200" value={form.current_product_stock} onChange={handleChange} />
              </div>
            </div>

            {/* Row 2 */}
            <div style={styles.grid2}>
              <div style={styles.field}>
                <label style={styles.label}>Product Category</label>
                <select style={styles.input} name="product_category" required
                  value={form.product_category} onChange={handleChange}>
                  <option value="">— select —</option>
                  {PRODUCT_CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div style={styles.field}>
                <label style={styles.label}>Product ID</label>
                <select style={styles.input} name="product_id" required
                  value={form.product_id} onChange={handleChange}>
                  <option value="">— select —</option>
                  {PRODUCTS.map((p) => <option key={p.id} value={p.id}>{p.id} ({p.name})</option>)}
                </select>
              </div>
            </div>

            {/* Row 3 — month and week_of_month are derived from date in Flask API */}
            <div style={styles.field}>
              <label style={styles.label}>Order Date</label>
              <input style={styles.input} type="date" name="order_date" required
                value={form.order_date} onChange={handleChange} />
            </div>

            {/* Submit */}
            <button type="submit" disabled={loading} style={styles.submitBtn}>
              {loading ? "Predicting …" : "Predict Quantity →"}
            </button>
          </form>

          {/* Result */}
          {result && (
            <div style={styles.resultBox}>
              <p style={styles.resultLabel}>Predicted Quantity Ordered</p>
              <p style={styles.resultNumber}>{result.predicted_quantity}</p>
              <p style={styles.resultSub}>
                Exact: {result.predicted_quantity_exact} · Scaled: {result.predicted_quantity_scaled}
              </p>
            </div>
          )}

          {/* Error */}
          {error && <div style={styles.errorBox}>⚠ {error}</div>}
        </div>

        <p style={styles.footer}>
          Flask API → https://inventory-predictor-api-webapp.onrender.com/api/predict
        </p>
      </div>
    </div>
  );
}