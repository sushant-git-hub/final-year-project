"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "leaflet.heat";

// Fix for default marker icon issues in Leaflet + Next.js
const DefaultIcon = L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-icon.png",
    shadowUrl: "https://unpkg.com/leaflet@1.7.1/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;

interface MapProps {
    center: [number, number];
    rankings: Array<{
        id: string;
        lat: number;
        lng: number;
        suitabilityScore: number;
        rank?: number;
    }>;
    heatmapData: Array<{
        lat: number;
        lng: number;
        intensity: number;
    }>;
}

export default function MapVisualization({ center, rankings, heatmapData }: MapProps) {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<L.Map | null>(null);

    useEffect(() => {
        if (!mapContainerRef.current) return;

        // Initialize map if not already done
        if (!mapInstanceRef.current) {
            mapInstanceRef.current = L.map(mapContainerRef.current).setView(center, 12);

            // Add tile layer (CartoDB Positron for a clean look)
            L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
                attribution:
                    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                subdomains: "abcd",
                maxZoom: 19,
            }).addTo(mapInstanceRef.current);
        } else {
            // Update center if it changes
            mapInstanceRef.current.setView(center);
        }

        const map = mapInstanceRef.current;

        // Clear existing layers (except tiles)
        map.eachLayer((layer) => {
            if (layer instanceof L.Marker || layer instanceof L.CircleMarker || (layer as any)._heat) {
                map.removeLayer(layer);
            }
        });

        // 1. Add Heatmap Layer
        if (heatmapData.length > 0) {
            const heatPoints = heatmapData.map((p) => [p.lat, p.lng, p.intensity]);
            (L as any).heatLayer(heatPoints, {
                radius: 25,
                blur: 15,
                maxZoom: 14,
                max: 1.0,
                gradient: {
                    0.4: "blue",
                    0.6: "cyan",
                    0.7: "lime",
                    0.8: "yellow",
                    1.0: "red",
                },
            }).addTo(map);
        }

        // 2. Add Markers for Rankings
        rankings.forEach((site, index) => {
            const isTopValues = index < 3;

            const marker = L.marker([site.lat, site.lng], {
                title: `Rank #${index + 1} (Score: ${site.suitabilityScore})`,
                zIndexOffset: 1000 - index, // Top ranks on top
            })
                .bindPopup(`
        <div style="font-family: sans-serif; min-width: 150px;">
            <strong style="color: #2563eb; font-size: 14px;">Rank #${index + 1}</strong>
            <div style="font-size: 12px; margin-top: 4px; color: #4b5563;">
                Hex: ${site.id}<br/>
                Score: <strong>${site.suitabilityScore}</strong>
            </div>
        </div>
      `)
                .addTo(map);

            // Open the popup for the #1 rank by default
            if (index === 0) {
                marker.openPopup();
            }
        });

        // Cleanup on unmount
        return () => {
            // We don't necessarily want to destroy the map instance on re-renders, 
            // but strict mode might cause issues. For now, we keep the instance alive.
        };
    }, [center, rankings, heatmapData]);

    return (
        <div className="relative h-full w-full rounded-xl overflow-hidden z-0">
            <div ref={mapContainerRef} className="h-full w-full" />

            {/* Legend Overlay */}
            <div className="absolute bottom-4 right-4 bg-white/90 backdrop-blur p-2 rounded shadow text-xs z-1000 border border-gray-200">
                <div className="font-semibold mb-1">Legend</div>
                <div className="flex items-center gap-2 mb-1">
                    <span className="w-3 h-3 rounded-full bg-blue-500 border border-white shadow-sm"></span>
                    <span>Recommended Site</span>
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-gradient-to-r from-blue-400 via-yellow-400 to-red-500 rounded-sm"></div>
                    <span>Demand Heatmap</span>
                </div>
            </div>
        </div>
    );
}
