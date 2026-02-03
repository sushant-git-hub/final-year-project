import LeafletHeatmap from "@/components/LeafletHeatMap";

export default function HeatmapSection({
  city,
}: {
  city: string | null;
}) {
  return (
    <div className="rounded-xl bg-white p-4 shadow">
      <h2 className="mb-2 text-xl font-semibold">
        Demand Heatmap & Coverage Zones
      </h2>

      <LeafletHeatmap city={city} />
    </div>
  );
}
