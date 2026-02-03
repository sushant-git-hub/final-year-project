"use client";

import { usePathname, useRouter } from "next/navigation";

const menuItems = [
  { label: "Dashboard", path: "/dashboard" },
  { label: "Results", path: "/results" },
  { label: "Reports", path: "/results" },
  { label: "Settings", path: "#" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <aside className="w-64 border-r bg-white p-4">
      <nav className="space-y-2">
        {menuItems.map((item) => (
          <button
            key={item.label}
            onClick={() => router.push(item.path)}
            className={`w-full rounded px-3 py-2 text-left text-sm ${
              pathname === item.path
                ? "bg-blue-100 text-blue-700"
                : "text-gray-700 hover:bg-gray-100"
            }`}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {/* Footer info */}
      <div className="mt-8 text-xs text-gray-500">
        ML-Based Retail Optimization
      </div>
    </aside>
  );
}
