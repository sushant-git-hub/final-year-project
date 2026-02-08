"use client";

import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend,
    ResponsiveContainer,
    ScatterChart,
    Scatter,
    PieChart,
    Pie,
    Cell,
} from "recharts";

interface ChartProps {
    rankings: Array<{
        id: string;
        suitabilityScore: number;
        demand: {
            demandScore: number;
            orderCount: number;
        };
        coverageScore: number;
        costScore: number;
        ward?: string;
        population?: number;
    }>;
}

const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884d8"];

export default function AnalysisCharts({ rankings }: ChartProps) {
    // 1. Prepare Data for Bar Chart (Top 5 Scores Breakdown)
    const barData = rankings.slice(0, 5).map((r, i) => ({
        name: `Rank #${i + 1}`,
        Demand: r.demand.demandScore * 100, // Scales for better visibility
        Coverage: r.coverageScore * 100,
        Cost: r.costScore * 100,
    }));

    // 2. Prepare Data for Scatter (Cost vs Coverage)
    const scatterData = rankings.map((r, i) => ({
        x: r.costScore,
        y: r.coverageScore,
        z: r.suitabilityScore, // bubble size?
        name: `Hex ${r.id.substring(0, 6)}...`,
        index: i,
    }));

    // 3. Prepare Data for Pie (Average Score Composition of Top 10)
    // Determine which factor contributes most on average
    const avgDemand =
        rankings.reduce((sum, r) => sum + r.demand.demandScore, 0) /
        rankings.length;
    const avgCoverage =
        rankings.reduce((sum, r) => sum + r.coverageScore, 0) / rankings.length;
    const avgCost =
        rankings.reduce((sum, r) => sum + r.costScore, 0) / rankings.length;

    const pieData = [
        { name: "Demand", value: avgDemand },
        { name: "Coverage", value: avgCoverage },
        { name: "Cost Efficiency", value: avgCost },
    ];

    return (
        <div className="space-y-8">
            {/* Chart Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Bar Chart */}
                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                    <h3 className="text-sm font-bold text-gray-700 uppercase mb-4">
                        Score Breakdown (Top 5)
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={barData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="name" fontSize={12} />
                                <YAxis fontSize={12} domain={[0, 100]} />
                                <Tooltip
                                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                />
                                <Legend iconType="circle" />
                                <Bar dataKey="Demand" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="Coverage" fill="#10b981" radius={[4, 4, 0, 0]} />
                                <Bar dataKey="Cost" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Scatter Chart */}
                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                    <h3 className="text-sm font-bold text-gray-700 uppercase mb-4">
                        Cost vs. Coverage Trade-off
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart
                                margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                            >
                                <CartesianGrid />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    name="Cost Score"
                                    label={{ value: 'Cost Efficiency →', position: 'bottom', offset: 0, fontSize: 12 }}
                                    domain={[0, 1]}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    name="Coverage Score"
                                    label={{ value: 'Coverage →', angle: -90, position: 'insideLeft', fontSize: 12 }}
                                    domain={[0, 1]}
                                />
                                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                                <Scatter name="Locations" data={scatterData} fill="#8884d8">
                                    {scatterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={index < 3 ? "#ef4444" : "#8884d8"} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-xs text-gray-400 mt-2 text-center">
                        * Red dots indicate Top 3 recommendations
                    </p>
                </div>
            </div>
            {/* Chart Row 1 - New Chart */}
            <div className="grid grid-cols-1 gap-6">
                {/* Population vs Suitability Chart */}
                <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm lg:col-span-2">
                    <h3 className="text-sm font-bold text-gray-700 uppercase mb-4">
                        Demographic Impact: Population vs. Suitability
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    type="number"
                                    dataKey="u"
                                    name="Population"
                                    unit=""
                                    domain={['auto', 'auto']}
                                    tickFormatter={(val) => (val / 1000).toFixed(0) + 'k'}
                                    label={{ value: 'Ward Population', position: 'bottom', offset: 0 }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="v"
                                    name="Suitability"
                                    domain={[0, 100]}
                                    label={{ value: 'Suitability Score', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                                <Scatter name="Wards" data={rankings.map(r => ({ u: r.population || 0, v: r.suitabilityScore, name: r.ward || 'Unknown' }))} fill="#82ca9d">
                                    {rankings.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={index < 3 ? "#ef4444" : "#82ca9d"} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-xs text-gray-400 mt-2 text-center">
                        * Higher population areas often correlate with higher demand but may have higher costs.
                    </p>
                </div>
            </div>

            {/* Chart Row 2 - Pie & Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="lg:col-span-1 bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                    <h3 className="text-sm font-bold text-gray-700 uppercase mb-4">
                        Average Factor Contribution
                    </h3>
                    <div className="h-[250px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie
                                    data={pieData}
                                    cx="50%"
                                    cy="50%"
                                    innerRadius={60}
                                    outerRadius={80}
                                    fill="#8884d8"
                                    paddingAngle={5}
                                    dataKey="value"
                                >
                                    {pieData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                    ))}
                                </Pie>
                                <Tooltip />
                                <Legend verticalAlign="bottom" height={36} />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="lg:col-span-2 bg-gradient-to-br from-indigo-50 to-blue-50 p-6 rounded-xl border border-blue-100 shadow-sm flex flex-col justify-center">
                    <h3 className="text-lg font-bold text-indigo-900 mb-2">
                        Analysis Report Summary
                    </h3>
                    <p className="text-indigo-800 text-sm mb-4 leading-relaxed">
                        Based on the analysis of <strong>{rankings.length}</strong> candidates, the top locations exhibit strong performance in
                        {' '}{avgDemand > avgCost ? 'Demand' : 'Cost Efficiency'}.
                        The recommended sites maximize the suitability index by balancing high order density with optimal coverage radii.
                    </p>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                        <div className="bg-white/60 p-3 rounded-lg">
                            <div className="text-xs text-indigo-500 uppercase font-semibold">Avg Suitability</div>
                            <div className="text-2xl font-bold text-indigo-900">
                                {Math.round(rankings.reduce((s, r) => s + r.suitabilityScore, 0) / rankings.length)}
                            </div>
                        </div>
                        <div className="bg-white/60 p-3 rounded-lg">
                            <div className="text-xs text-indigo-500 uppercase font-semibold">Max Demand</div>
                            <div className="text-2xl font-bold text-indigo-900">
                                {Math.max(...rankings.map(r => r.demand.demandScore)).toFixed(2)}
                            </div>
                        </div>
                        <div className="bg-white/60 p-3 rounded-lg">
                            <div className="text-xs text-indigo-500 uppercase font-semibold">Top Candidate</div>
                            <div className="text-lg font-bold text-indigo-900 truncate" title={rankings[0]?.id}>
                                {rankings[0]?.id.substring(0, 8)}...
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={() => alert("Printing Report PDF... (Feature Mock)")}
                        className="self-start px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition shadow-sm flex items-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        Download Full Report
                    </button>
                </div>
            </div>
        </div>
    );
}
