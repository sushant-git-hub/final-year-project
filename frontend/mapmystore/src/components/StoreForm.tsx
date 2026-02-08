"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function StoreForm() {
  const router = useRouter();

  const [city, setCity] = useState("");
  const [type, setType] = useState("");
  const [budget, setBudget] = useState("");
  const [radius, setRadius] = useState("");
  const [income, setIncome] = useState("");
  const [zone, setZone] = useState("");
  const [proximity, setProximity] = useState<string[]>([]);

  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // Toggle proximity selection
  const toggleProximity = (value: string) => {
    setProximity((prev) =>
      prev.includes(value) ? prev.filter((p) => p !== value) : [...prev, value]
    );
  };

  const validateInputs = () => {
    if (!city || !type || !budget || !radius) {
      setError("City, Type, Budget and Radius are required.");
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

    // Encode all params
    const params = new URLSearchParams({
      city,
      type,
      budget,
      radius,
      income,
      zone,
      proximity: proximity.join(","), // Pass as comma-separated string
    });

    // Simulate system processing (data collection + ML analysis)
    setTimeout(() => {
      router.push(`/results?${params.toString()}`);
    }, 800);
  };

  return (
    <div className="rounded-xl bg-white p-6 shadow-sm border border-gray-100">
      <h2 className="mb-6 text-xl font-semibold text-gray-800 border-b border-gray-50 pb-2">
        Store Configuration
      </h2>

      {/* Error Message */}
      {error && (
        <div className="mb-4 rounded bg-red-50 px-3 py-2 text-sm text-red-600 border border-red-100">
          {error}
        </div>
      )}

      <div className="space-y-6">
        {/* Row 1: Basic Info */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">City</label>
            <input
              placeholder="e.g. pune"
              className="w-full rounded-lg border border-gray-200 p-2 text-sm focus:border-blue-500 focus:ring-1 outline-none transition-all"
              value={city}
              onChange={(e) => setCity(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">Store Type</label>
            <select
              className="w-full rounded-lg border border-gray-200 p-2 text-sm focus:border-blue-500 focus:ring-1 outline-none bg-white transition-all"
              value={type}
              onChange={(e) => setType(e.target.value)}
            >
              <option value="">Select option</option>
              <option value="retail">Grocery / Retail</option>
              <option value="pharmacy">Pharmacy</option>
              <option value="clothing">Clothing / Apparel</option>
              <option value="cafe">Cafe / Restaurant</option>
              <option value="logistics">Logistics Hub</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">Budget (₹)</label>
            <input
              type="number"
              placeholder="e.g. 500000"
              className="w-full rounded-lg border border-gray-200 p-2 text-sm focus:border-blue-500 focus:ring-1 outline-none transition-all"
              value={budget}
              onChange={(e) => setBudget(e.target.value)}
            />
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">Radius (km)</label>
            <input
              type="number"
              placeholder="1-20"
              className="w-full rounded-lg border border-gray-200 p-2 text-sm focus:border-blue-500 focus:ring-1 outline-none transition-all"
              value={radius}
              onChange={(e) => setRadius(e.target.value)}
            />
          </div>
        </div>

        {/* Row 2: Advanced Targeting */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">Income Level (Optional)</label>
            <select
              className="w-full rounded-lg border border-gray-200 p-2 text-sm focus:border-blue-500 focus:ring-1 outline-none bg-white transition-all"
              value={income}
              onChange={(e) => setIncome(e.target.value)}
            >
              <option value="">Any</option>
              <option value="low">Low Income</option>
              <option value="middle">Middle Income</option>
              <option value="high">High Income</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">Preferred Zone (Optional)</label>
            <select
              className="w-full rounded-lg border border-gray-200 p-2 text-sm focus:border-blue-500 focus:ring-1 outline-none bg-white transition-all"
              value={zone}
              onChange={(e) => setZone(e.target.value)}
            >
              <option value="">Any</option>
              <option value="north">North</option>
              <option value="south">South</option>
              <option value="east">East</option>
              <option value="west">West</option>
              <option value="central">Central</option>
            </select>
          </div>

          <div className="space-y-1">
            <label className="text-xs font-medium text-gray-500 uppercase">Proximity (Optional)</label>
            <div className="flex flex-wrap gap-2 mt-1">
              {['Metro', 'Mall', 'Highway'].map(p => (
                <button
                  key={p}
                  onClick={() => toggleProximity(p)}
                  className={`px-3 py-1 text-xs rounded-full border transition-all ${proximity.includes(p)
                    ? 'bg-blue-100 border-blue-200 text-blue-700 font-medium'
                    : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                    }`}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>

        </div>
      </div>

      {/* Submit Button */}
      <div className="mt-8 flex justify-end border-t border-gray-50 pt-4">
        <button
          onClick={handleSubmit}
          disabled={loading}
          className={`rounded-lg px-8 py-3 text-sm font-bold text-white transition-all shadow-md transform active:scale-95 ${loading
            ? "cursor-not-allowed bg-gray-400"
            : "bg-blue-600 hover:bg-blue-700 hover:shadow-lg"
            }`}
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></span>
              Analyzing 40+ ML Features...
            </span>
          ) : "Run AI Analysis"}
        </button>
      </div>
    </div>
  );
}
