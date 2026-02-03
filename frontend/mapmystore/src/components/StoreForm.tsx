"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function StoreForm() {
  const router = useRouter();

  const [city, setCity] = useState("");
  const [type, setType] = useState("");
  const [budget, setBudget] = useState("");
  const [radius, setRadius] = useState("");

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const validateInputs = () => {
    if (!city || !type || !budget || !radius) {
      setError("All fields are required.");
      return false;
    }

    if (Number(budget) <= 0) {
      setError("Budget must be greater than zero.");
      return false;
    }

    if (Number(radius) < 1 || Number(radius) > 20) {
      setError("Delivery radius must be between 1 and 20 km.");
      return false;
    }

    return true;
  };

  const handleSubmit = () => {
    setError("");

    if (!validateInputs()) return;

    setLoading(true);

    // Simulate system processing (data collection + ML analysis)
    setTimeout(() => {
      router.push(
        `/results?city=${city}&type=${type}&budget=${budget}&radius=${radius}`
      );
    }, 1200);
  };

  return (
    <div className="rounded-xl bg-white p-4 shadow">
      <h2 className="mb-4 text-xl font-semibold">
        Store Configuration
      </h2>

      {/* Error Message */}
      {error && (
        <div className="mb-4 rounded bg-red-100 px-3 py-2 text-sm text-red-700">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-4">
        {/* City */}
        <input
          placeholder="City"
          className="rounded border p-2"
          value={city}
          onChange={(e) => setCity(e.target.value)}
        />

        {/* Store Type */}
        <select
          className="rounded border p-2"
          value={type}
          onChange={(e) => setType(e.target.value)}
        >
          <option value="retail">Retail / Quick Commerce</option>
          <option value="franchise">Franchise Planning</option>
          <option value="logistics">Logistics / Warehousing</option>
          <option value="urban">Urban Planning</option>
          <option value="investment">Investment Analysis</option>
          <option value="consulting">Consulting / Strategy</option>
        </select>

        {/* Budget */}
        <input
          type="number"
          placeholder="Budget"
          className="rounded border p-2"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
        />

        {/* Radius */}
        <input
          type="number"
          placeholder="Delivery Radius (km)"
          className="rounded border p-2"
          value={radius}
          onChange={(e) => setRadius(e.target.value)}
        />
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        className={`mt-4 rounded px-4 py-2 text-white ${loading
            ? "cursor-not-allowed bg-gray-400"
            : "bg-green-600 hover:bg-green-700"
          }`}
      >
        {loading ? "Analyzing Locations..." : "Generate Recommendations"}
      </button>
    </div>
  );
}
