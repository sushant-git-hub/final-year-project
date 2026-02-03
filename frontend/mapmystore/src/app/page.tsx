"use client";

import { useRouter } from "next/navigation";

export default function HomePage() {
  const router = useRouter();

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-100 px-4">
      <div className="max-w-xl text-center">
        <h1 className="mb-4 text-4xl font-bold text-gray-800">
          MapMyStore
        </h1>

        <p className="mb-6 text-gray-600">
          Machine Learning–Based Retail Site Optimization System
        </p>

        <div className="flex justify-center gap-4">
          <button
            onClick={() => router.push("/login")}
            className="rounded bg-blue-600 px-6 py-2 text-white hover:bg-blue-700"
          >
            Login
          </button>

          <button
            onClick={() => router.push("/dashboard")}
            className="rounded bg-gray-800 px-6 py-2 text-white hover:bg-gray-900"
          >
            Demo Dashboard
          </button>
        </div>
      </div>
    </main>
  );
}
