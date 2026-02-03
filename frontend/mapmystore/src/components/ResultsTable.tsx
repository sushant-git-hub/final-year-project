type ResultRow = {
  rank: number;
  location: string;
  demand: number;
  cost: number;
  score: number;
  roi: "High" | "Medium" | "Low";
  delivery: "Fast" | "Moderate" | "Slow";
};

const data: ResultRow[] = [
  {
    rank: 1,
    location: "Zone A",
    demand: 85,
    cost: 60,
    score: 90,
    roi: "High",
    delivery: "Fast",
  },
  {
    rank: 2,
    location: "Zone B",
    demand: 78,
    cost: 55,
    score: 82,
    roi: "Medium",
    delivery: "Moderate",
  },
  {
    rank: 3,
    location: "Zone C",
    demand: 70,
    cost: 50,
    score: 75,
    roi: "Medium",
    delivery: "Moderate",
  },
];

// Utility functions
const scoreColor = (score: number) => {
  if (score >= 85) return "bg-green-500";
  if (score >= 70) return "bg-yellow-500";
  return "bg-red-500";
};

const badgeStyle = (value: string) => {
  switch (value) {
    case "High":
    case "Fast":
      return "bg-green-100 text-green-700";
    case "Medium":
    case "Moderate":
      return "bg-yellow-100 text-yellow-700";
    default:
      return "bg-red-100 text-red-700";
  }
};

export default function ResultsTable() {
  return (
    <div className="rounded-xl bg-white p-5 shadow">
      {/* Header */}
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h2 className="text-xl font-semibold">
            Ranked Location Recommendations
          </h2>
          <p className="text-sm text-gray-500">
            Ranked based on demand, cost feasibility, delivery efficiency, and
            profitability
          </p>
        </div>

        {/* Legend */}
        <div className="flex gap-2 text-xs">
          <span className="rounded bg-green-100 px-2 py-1 text-green-700">
            High
          </span>
          <span className="rounded bg-yellow-100 px-2 py-1 text-yellow-700">
            Medium
          </span>
          <span className="rounded bg-red-100 px-2 py-1 text-red-700">
            Low
          </span>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b bg-gray-50 text-left text-gray-600">
              <th className="p-3">Rank</th>
              <th className="p-3">Location</th>
              <th className="p-3">Demand</th>
              <th className="p-3">Cost</th>
              <th className="p-3">ROI</th>
              <th className="p-3">Delivery</th>
              <th className="p-3">Suitability</th>
            </tr>
          </thead>

          <tbody>
            {data.map((row) => (
              <tr
                key={row.rank}
                className={`border-b hover:bg-gray-50 ${
                  row.rank === 1 ? "bg-green-50" : ""
                }`}
              >
                {/* Rank */}
                <td className="p-3 font-semibold">
                  #{row.rank}
                </td>

                {/* Location */}
                <td className="p-3 font-medium">
                  {row.location}
                  {row.rank === 1 && (
                    <span className="ml-2 rounded bg-green-600 px-2 py-0.5 text-xs text-white">
                      Best Choice
                    </span>
                  )}
                </td>

                {/* Demand */}
                <td className="p-3">{row.demand}</td>

                {/* Cost */}
                <td className="p-3">{row.cost}</td>

                {/* ROI */}
                <td className="p-3">
                  <span
                    className={`rounded px-2 py-1 text-xs font-semibold ${badgeStyle(
                      row.roi
                    )}`}
                  >
                    {row.roi} ROI
                  </span>
                </td>

                {/* Delivery Efficiency */}
                <td className="p-3">
                  <span
                    className={`rounded px-2 py-1 text-xs font-semibold ${badgeStyle(
                      row.delivery
                    )}`}
                  >
                    {row.delivery}
                  </span>
                </td>

                {/* Suitability Score */}
                <td className="p-3">
                  <div className="flex items-center gap-3">
                    <span className="font-semibold">
                      {row.score}
                    </span>

                    <div className="h-2 w-28 rounded bg-gray-200">
                      <div
                        className={`h-2 rounded ${scoreColor(row.score)}`}
                        style={{ width: `${row.score}%` }}
                      />
                    </div>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Footer Note */}
      <p className="mt-3 text-xs text-gray-500">
        * Rankings are generated using predictive demand modeling, cost
        feasibility analysis, and delivery efficiency metrics.
      </p>
    </div>
  );
}
