"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";

// 🔧 Fix marker icon paths (required for Next.js)
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "/leaflet/marker-icon-2x.png",
  iconUrl: "/leaflet/marker-icon.png",
  shadowUrl: "/leaflet/marker-shadow.png",
});

export default function HeatmapSection({ city }: { city: string | null }) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<L.Map | null>(null);

  useEffect(() => {
    if (!mapRef.current || !city) return;

    // 🔥 IMPORTANT: destroy existing map FIRST
    if (mapInstance.current) {
      mapInstance.current.remove();
      mapInstance.current = null;
    }

    // Create map
    const map = L.map(mapRef.current).setView(
      [28.6139, 77.2090], // demo center
      11
    );

    mapInstance.current = map;

    // Tiles
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    // Demo heatmap
    const heatData: [number, number, number][] = [
      [28.6448, 77.2167, 0.9],
      [28.6290, 77.1910, 0.7],
      [28.6120, 77.2410, 0.8],
    ];

    (L as any).heatLayer(heatData, {
      radius: 30,
      blur: 20,
    }).addTo(map);

    // Coverage radius
    L.circle([28.6139, 77.2090], {
      radius: 5000,
      color: "#2563eb",
      fillOpacity: 0.15,
    }).addTo(map);

    return () => {
      // Cleanup on unmount
      if (mapInstance.current) {
        mapInstance.current.remove();
        mapInstance.current = null;
      }
    };
  }, [city]); // rerun only when city changes

  return (
    <div className="rounded-xl bg-white shadow">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">Demand Heatmap</h2>
        <span className="rounded bg-yellow-100 px-2 py-1 text-xs text-yellow-800">
          Demo
        </span>
      </div>

      <div className="h-[300px]">
        <div ref={mapRef} className="h-full w-full" />
      </div>
    </div>
  );
}
