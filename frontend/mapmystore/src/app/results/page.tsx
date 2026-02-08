"use client";

import { useSearchParams } from "next/navigation";
import { useEffect, useState, Suspense } from "react";
import Link from "next/link";
import dynamic from "next/dynamic";

import AnalysisCharts from "@/components/AnalysisCharts";
import ExportPanel from "@/components/ExportPanel";
import ResultsTable from "@/components/ResultsTable";

// Dynamically load Leaflet map (no SSR)
const MapVisualization = dynamic(
  () => import("@/components/MapVisualization"),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full w-full items-center justify-center bg-gray-100 text-gray-400">
        Loading map…
      </div>
    ),
  }
);

// =====================
// Types
// =====================
interface SiteAnalysisResponse {
  meta: {
    city: string;
    candidateCount: number;
    timestamp: string;
  };
  rankings: Array<{
    id: string;
    lat: number;
    lng: number;

    // Core ML Outputs
    suitabilityScore: number;
    successScore: number;
    successProbability: number;
    expectedRevenue: number;

    // Feature Breakdown
    demand: {
      demandScore: number;
      orderCount: number;
    };
    coverageScore: number;
    costScore: number;

    // Rich Metadata
    ward?: string;
    population?: number;
    keyFactors: string[];
    metrics: {
      rent: number;
      footfall: number;
      competition: string;
    }
  }>;
  heatmap?: Array<{
    lat: number;
    lng: number;
    intensity: number;
  }>;
}

// =====================
// Page Content
// =====================
function ResultsContent() {
  const searchParams = useSearchParams();

  const city = searchParams.get("city");
  const type = searchParams.get("type");
  const budget = searchParams.get("budget");
  const radius = searchParams.get("radius");
  const income = searchParams.get("income");
  const zone = searchParams.get("zone");
  const proximity = searchParams.get("proximity");

  const [data, setData] = useState<SiteAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // =====================
  // Fetch backend data
  // =====================
  useEffect(() => {
    if (!city) return;

    const fetchData = async () => {
      try {
        const res = await fetch("http://localhost:3005/api/analyze-site", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            city,
            type,
            budget: Number(budget),
            radius: Number(radius),
            income,
            zone,
            proximity: proximity ? proximity.split(',') : []
          }),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.message || "Failed to fetch analysis");
        }

        const result = await res.json();
        setData(result);
      } catch (err: any) {
        setError(err.message || "Unexpected error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [city, type, budget, radius, income, zone, proximity]);

  // =====================
  // States
  // =====================
  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-4 border-blue-600 border-t-transparent" />
          <h3 className="text-lg font-semibold text-gray-800">
            Analyzing {city}'s Ecosystem
          </h3>
          <p className="mt-2 text-sm text-gray-500">
            Evaluating 40+ ML features including footfall, demographics, and competition...
          </p>
        </div>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-50">
        <div className="max-w-md rounded-xl border border-red-100 bg-white p-6 text-center shadow">
          <h2 className="mb-2 text-lg font-bold text-gray-800">
            Analysis Failed
          </h2>
          <p className="mb-6 text-red-600">{error}</p>
          <Link
            href="/"
            className="rounded-lg bg-gray-900 px-5 py-2 text-white hover:bg-gray-800"
          >
            Try Again
          </Link>
        </div>
      </div>
    );
  }

  // =====================
  // Render
  // =====================
  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold">
              AI Site Recommendations for {data.meta.city}
            </h1>
            <p className="text-sm text-gray-500">
              Analyzed {data.meta.candidateCount} locations based on your {type} store profile
            </p>
          </div>

          <ExportPanel />
        </div>

        {/* Charts */}
        <div className="mb-10">
          <h2 className="mb-4 text-xl font-bold">Market Intelligence</h2>
          <AnalysisCharts rankings={data.rankings} />
        </div>

        {/* Rankings + Map */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Rankings List */}
          <div className="space-y-4">
            {data.rankings.map((site, index) => (
              <div
                key={site.id}
                className={`rounded-xl border bg-white p-5 transition hover:shadow-md ${index === 0
                    ? "border-blue-400 ring-1 ring-blue-400"
                    : "border-gray-200"
                  }`}
              >
                {/* Top Row: Score & Revenue */}
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className={`flex h-16 w-16 flex-col items-center justify-center rounded-2xl ${site.successScore >= 80 ? 'bg-green-100 text-green-700' :
                        site.successScore >= 60 ? 'bg-yellow-100 text-yellow-700' : 'bg-red-100 text-red-700'
                      }`}>
                      <span className="text-2xl font-bold">{site.successScore}</span>
                      <span className="text-[10px] font-medium uppercase">Score</span>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500 uppercase font-semibold">
                        Est. Monthly Revenue
                      </div>
                      <div className="text-xl font-bold text-gray-900">
                        ₹{site.expectedRevenue.toLocaleString()}
                      </div>
                      <div className="text-xs text-green-600 font-medium">
                        {site.successProbability}% Success Probability
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="inline-block rounded-full bg-blue-50 px-3 py-1 text-xs font-bold text-blue-700">
                      Rank #{index + 1}
                    </div>
                  </div>
                </div>

                {/* Middle Row: Key Factors */}
                <div className="mb-4 flex flex-wrap gap-2">
                  {site.keyFactors.map(factor => (
                    <span key={factor} className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded border border-gray-200">
                      {factor}
                    </span>
                  ))}
                </div>

                {/* Bottom Row: Detailed Metrics */}
                <div className="grid grid-cols-3 gap-4 border-t pt-4 text-sm bg-gray-50/50 -mx-5 -mb-5 p-4 rounded-b-xl">
                  <div>
                    <div className="text-xs text-gray-400">Location</div>
                    <div className="font-medium text-gray-800 truncate" title={site.ward}>
                      {site.ward || 'Unknown'}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Rent Estimate</div>
                    <div className="font-medium text-gray-800">
                      ₹{site.metrics.rent.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-400">Footfall/Day</div>
                    <div className="font-medium text-gray-800">
                      ~{site.metrics.footfall.toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Map */}
          <div className="sticky top-6 h-[600px] rounded-xl border bg-white shadow-sm overflow-hidden">
            <MapVisualization
              center={[data.rankings[0].lat, data.rankings[0].lng]}
              rankings={data.rankings}
              heatmapData={data.heatmap ?? []}
            />
          </div>
        </div>

        {/* Optional table (demo or refactor later) */}
        <div className="mt-10">
          <ResultsTable />
        </div>
      </div>
    </div>
  );
}

// =====================
// Score badge
// =====================
function ScoreBadge({ label, value }: { label: string; value: number }) {
  let color = "bg-red-100 text-red-700";
  if (value >= 0.7) color = "bg-green-100 text-green-700";
  else if (value >= 0.4) color = "bg-yellow-100 text-yellow-700";

  return (
    <div className={`rounded px-2 py-1 text-center text-xs ${color}`}>
      <div>{label}</div>
      <div className="font-bold">{value.toFixed(2)}</div>
    </div>
  );
}

// =====================
// Page wrapper
// =====================
export default function ResultsPage() {
  return (
    <Suspense fallback={<div>Loading…</div>}>
      <ResultsContent />
    </Suspense>
  );
}