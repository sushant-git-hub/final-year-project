"use client";

import { useSearchParams } from "next/navigation";
import HeatmapSection from "@/components/HeatmapSection";
import ResultsTable from "@/components/ResultsTable";
import ChartsSection from "@/components/ChartsSection";
import ExportPanel from "@/components/ExportPanel";

export default function ResultsPage() {
  const params = useSearchParams();
  const city = params.get("city");

  return (
    <main className="min-h-screen bg-gray-100 p-6">
      {/* Header */}
      <header className="mb-6">
        <h1 className="text-3xl font-bold">
          Retail Site Recommendation Results
        </h1>
        <p className="mt-1 text-gray-600">
          Analysis for <span className="font-medium">{city}</span>
        </p>
      </header>

      {/* Export Bar */}
      <section className="mb-4 flex items-center justify-between rounded-lg bg-white p-4 shadow">
        <div>
          <h2 className="text-lg font-semibold">
            Location Analysis Overview
          </h2>
          <p className="text-sm text-gray-500">
            Review top-ranked locations and export reports
          </p>
        </div>

        <ExportPanel />
      </section>

      {/* Map */}
      <section className="mb-6">
        <HeatmapSection city={city} />
      </section>

      {/* ⭐ MOST IMPORTANT: Ranked Table */}
      <section className="mb-10">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-xl font-semibold">
            Top Recommended Locations
          </h2>
          <span className="text-sm text-blue-600">
            Sorted by suitability score
          </span>
        </div>

        <ResultsTable />
      </section>

      {/* Secondary: Charts */}
      <section>
        <div className="mb-3">
          <h2 className="text-xl font-semibold">
            Analytical Insights
          </h2>
          <p className="text-sm text-gray-500">
            Supporting charts for demand, cost, and suitability analysis
          </p>
        </div>

        <ChartsSection />
      </section>
    </main>
  );
}
