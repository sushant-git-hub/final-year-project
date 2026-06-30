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
        successScore: number;
        successProbability: number;
        expectedRevenue: number;
        confidenceLevel?: string;
        recommendation?: string;
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
const CONFIDENCE_COLORS: Record<string, string> = {
    'VERY_HIGH': '#10b981',
    'HIGH': '#34d399',
    'MEDIUM': '#fbbf24',
    'LOW': '#fb923c',
    'VERY_LOW': '#ef4444'
};

export default function AnalysisCharts({ rankings }: ChartProps) {
    // 1. Prepare Data for Bar Chart (Top 5 Estimated Revenues)
    const barData = rankings.slice(0, 5).map((r, i) => ({
        name: `Rank #${i + 1}`,
        Revenue: r.expectedRevenue,
        Score: r.successScore,
    }));

    // 2. Prepare Data for Scatter (Revenue vs Probability)
    const scatterData = rankings.map((r, i) => ({
        x: r.successProbability,
        y: r.expectedRevenue,
        z: r.successScore, // bubble size indicator
        name: `Rank #${i + 1} (${r.ward || 'Hex'})`,
        index: i,
    }));

    // 3. Prepare Data for Pie (Confidence Level Distribution)
    const confidenceCounts: Record<string, number> = {};
    rankings.forEach(r => {
        const lvl = r.confidenceLevel || 'UNKNOWN';
        confidenceCounts[lvl] = (confidenceCounts[lvl] || 0) + 1;
    });
    
    const pieData = Object.entries(confidenceCounts).map(([name, value]) => ({
        name: name.replace('_', ' '),
        value,
        originalName: name
    }));

    // Summaries
    const topRevenue = Math.max(0, ...rankings.map(r => r.expectedRevenue || 0));
    const avgScore = rankings.reduce((sum, r) => sum + r.successScore, 0) / (rankings.length || 1);
    const avgProb = rankings.reduce((sum, r) => sum + r.successProbability, 0) / (rankings.length || 1);

    return (
        <div className="space-y-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Bar Chart: Revenue Projection */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm transition-colors">
                    <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase mb-4">
                        Revenue Projection (Top 5 Candidates)
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={barData}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                <XAxis dataKey="name" fontSize={12} />
                                <YAxis fontSize={12} tickFormatter={(value) => `₹${value/1000}k`} />
                                <Tooltip
                                    formatter={(value: any) => [`₹${Number(value).toLocaleString('en-IN')}`, 'Estimated Revenue']}
                                    contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                />
                                <Bar dataKey="Revenue" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Scatter Chart: Probability vs Revenue */}
                <div className="bg-white dark:bg-gray-800 p-6 rounded-xl border border-gray-200 dark:border-gray-700 shadow-sm transition-colors">
                    <h3 className="text-sm font-bold text-gray-700 dark:text-gray-300 uppercase mb-4">
                        Success Probability vs Expected Revenue
                    </h3>
                    <div className="h-[300px] w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 40 }}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    type="number"
                                    dataKey="x"
                                    name="Success Prob"
                                    unit="%"
                                    domain={[0, 100]}
                                    label={{ value: 'Success Probability (%) →', position: 'bottom', offset: 0, fontSize: 12 }}
                                />
                                <YAxis
                                    type="number"
                                    dataKey="y"
                                    name="Est. Revenue"
                                    tickFormatter={(val) => `₹${val/1000}k`}
                                    label={{ value: 'Expected Revenue (₹) →', angle: -90, position: 'insideLeft', offset: 10, fontSize: 12, dx: -40 }}
                                />
                                <Tooltip 
                                    cursor={{ strokeDasharray: "3 3" }} 
                                    formatter={(value: any, name: any) => [name === 'Est. Revenue' ? `₹${Number(value).toLocaleString('en-IN')}` : `${value}%`, name]}
                                />
                                <Scatter name="Candidates" data={scatterData}>
                                    {scatterData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={index < 3 ? "#ef4444" : "#8884d8"} />
                                    ))}
                                </Scatter>
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                    <p className="text-xs text-gray-400 mt-2 text-center">
                        * Red dots indicate Top 3 recommended sites
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Summary Panel */}
                <div className="lg:col-span-3 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-gray-800 dark:to-gray-700 p-6 rounded-xl border border-blue-100 dark:border-gray-600 shadow-sm flex flex-col justify-center transition-colors">
                    <h3 className="text-lg font-bold text-indigo-900 dark:text-indigo-100 mb-2">
                        XGBoost Insights Report
                    </h3>
                    <p className="text-indigo-800 dark:text-indigo-200 text-sm mb-4 leading-relaxed">
                        The ML model evaluated <strong>{rankings.length}</strong> viable candidates. The highest potential revenue identified is 
                        <strong> ₹{topRevenue.toLocaleString('en-IN')}</strong> per month. The candidates exhibit an average success probability of 
                        <strong> {Math.round(avgProb)}%</strong> based on the target demographic.
                    </p>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                        <div className="bg-white/60 dark:bg-gray-900/40 p-3 rounded-lg border border-transparent dark:border-gray-800/50">
                            <div className="text-xs text-indigo-500 dark:text-indigo-400 uppercase font-semibold">Avg Success Score</div>
                            <div className="text-2xl font-bold text-indigo-900 dark:text-indigo-100">
                                {Math.round(avgScore)}/100
                            </div>
                        </div>
                        <div className="bg-white/60 dark:bg-gray-900/40 p-3 rounded-lg border border-transparent dark:border-gray-800/50">
                            <div className="text-xs text-indigo-500 dark:text-indigo-400 uppercase font-semibold">Max Revenue</div>
                            <div className="text-2xl font-bold text-green-700 dark:text-green-500">
                                ₹{(topRevenue / 1000).toFixed(1)}k
                            </div>
                        </div>
                        <div className="bg-white/60 dark:bg-gray-900/40 p-3 rounded-lg border border-transparent dark:border-gray-800/50">
                            <div className="text-xs text-indigo-500 dark:text-indigo-400 uppercase font-semibold">Top Location</div>
                            <div className="text-lg font-bold text-indigo-900 dark:text-indigo-100 truncate" title={rankings[0]?.ward}>
                                {rankings[0]?.ward || "Unknown Ward"}
                            </div>
                        </div>
                    </div>

                    <button
                        onClick={() => alert("Downloading Report... (Feature Mock)")}
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
