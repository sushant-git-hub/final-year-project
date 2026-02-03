"use client";

import { useEffect, useRef, useState } from "react";
import "leaflet/dist/leaflet.css";

type Props = {
  city: string | null;
};

export default function LeafletHeatmap({ city }: Props) {
  const mapRef = useRef<HTMLDivElement | null>(null);
  const mapInstance = useRef<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!mapRef.current || !city) return;

    const loadMap = async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet.heat");

      // 🔍 Geocode city (FREE)
      const res = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${city}`
      );
      const data = await res.json();

      if (!data || data.length === 0) return;

      const lat = parseFloat(data[0].lat);
      const lon = parseFloat(data[0].lon);

      // 🧹 Cleanup previous map
      if (mapInstance.current) {
        mapInstance.current.remove();
      }

      // 🗺️ Create map
      mapInstance.current = L.map(mapRef.current).setView(
        [lat, lon],
        12
      );

      // 🌍 Tiles
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(mapInstance.current);

      // 🔥 DEMO HEATMAP POINTS
      const heatPoints = Array.from({ length: 20 }).map(() => [
        lat + (Math.random() - 0.5) * 0.1,
        lon + (Math.random() - 0.5) * 0.1,
        Math.random(),
      ]);

      (L as any).heatLayer(heatPoints, {
        radius: 30,
        blur: 20,
        maxZoom: 13,
      }).addTo(mapInstance.current);

      // 🔵 Coverage circle (5km)
      L.circle([lat, lon], {
        radius: 5000,
        color: "#2563eb",
        fillColor: "#3b82f6",
        fillOpacity: 0.15,
      }).addTo(mapInstance.current);

      setLoading(false);
    };

    loadMap();

    return () => {
      if (mapInstance.current) {
        mapInstance.current.remove();
      }
    };
  }, [city]);

  return (
    <div className="relative">
      {loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded bg-gray-100 text-gray-600">
          Loading map for <strong className="ml-1">{city}</strong>…
        </div>
      )}

      <div
        ref={mapRef}
        className="h-[450px] w-full rounded-lg"
      />
    </div>
  );
}
