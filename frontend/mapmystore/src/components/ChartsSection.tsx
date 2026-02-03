"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  Legend,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  ReferenceLine,
} from "recharts";

const data = [
  { name: "Zone A", demand: 85, cost: 60, score: 90 },
  { name: "Zone B", demand: 78, cost: 55, score: 82 },
  { name: "Zone C", demand: 70, cost: 50, score: 75 },
];

const COLORS = ["#2563eb", "#16a34a", "#f97316"];

export default function ChartsSection() {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
      {/* 1️⃣ Demand vs Cost */}
      <div className="h-[330px] rounded-xl bg-white p-4 shadow">
        <h3 className="mb-1 text-sm font-semibold">
          Demand vs Cost
        </h3>
        <p className="mb-2 text-xs text-gray-500">
          Higher demand with controlled cost indicates profitability
        </p>

        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis
                label={{
                  value: "Score (0–100)",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <Tooltip
                formatter={(value, name) =>
                  name === "demand"
                    ? [`${value}`, "Demand Score"]
                    : [`${value}`, "Cost Score"]
                }
              />
              <Legend />
              <ReferenceLine
                y={70}
                stroke="#9ca3af"
                strokeDasharray="3 3"
                label="Min Acceptable"
              />
              <Bar dataKey="demand" fill="#2563eb" name="Demand" />
              <Bar dataKey="cost" fill="#f97316" name="Cost" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 2️⃣ Suitability Score */}
      <div className="h-[330px] rounded-xl bg-white p-4 shadow">
        <h3 className="mb-1 text-sm font-semibold">
          Suitability Score
        </h3>
        <p className="mb-2 text-xs text-gray-500">
          Overall site ranking based on all evaluation factors
        </p>

        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis
                domain={[0, 100]}
                label={{
                  value: "Suitability Score",
                  angle: -90,
                  position: "insideLeft",
                }}
              />
              <Tooltip
                formatter={(value) => [`${value}`, "Suitability"]}
              />
              <Legend />
              <ReferenceLine
                y={75}
                stroke="#16a34a"
                strokeDasharray="4 4"
                label="Recommended Threshold"
              />
              <Line
                type="monotone"
                dataKey="score"
                stroke="#16a34a"
                strokeWidth={3}
                dot={{ r: 4 }}
                name="Suitability Score"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3️⃣ Score Distribution */}
      <div className="h-[330px] rounded-xl bg-white p-4 shadow">
        <h3 className="mb-1 text-sm font-semibold">
          Suitability Distribution
        </h3>
        <p className="mb-2 text-xs text-gray-500">
          Proportional comparison of candidate locations
        </p>

        <div className="h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Tooltip
                formatter={(value) => [`${value}`, "Score"]}
              />
              <Legend />
              <Pie
                data={data}
                dataKey="score"
                nameKey="name"
                outerRadius={85}
                label
              >
                {data.map((_, index) => (
                  <Cell
                    key={index}
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
