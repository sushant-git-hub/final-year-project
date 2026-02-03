"use client";

import { useRouter } from "next/navigation";

export default function Header() {
  const router = useRouter();

  return (
    <header className="flex h-14 items-center justify-between border-b bg-white px-6 shadow-sm">
      {/* Left */}
      <div className="font-bold text-lg text-blue-600">
        MapMyStore
      </div>

      {/* Right */}
      <div className="flex items-center gap-4">
        <span className="text-sm text-gray-600">
          Retail Manager
        </span>

        <button
          onClick={() => router.push("/login")}
          className="rounded bg-red-500 px-3 py-1 text-sm text-white hover:bg-red-600"
        >
          Logout
        </button>
      </div>
    </header>
  );
}
