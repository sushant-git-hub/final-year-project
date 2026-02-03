"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const validateInputs = () => {
    if (!email || !password) {
      setError("All fields are required.");
      return false;
    }

    if (!email.includes("@")) {
      setError("Please enter a valid email address.");
      return false;
    }

    return true;
  };

  const handleLogin = async () => {
    setError("");

    if (!validateInputs()) return;

    setLoading(true);

    // 🔹 Dummy authentication (demo)
    setTimeout(() => {
      if (email === "admin@mapmystore.com" && password === "admin123") {
        router.push("/dashboard");
      } else {
        setError("Invalid email or password.");
      }
      setLoading(false);
    }, 1200);
  };

  return (
    <main className="flex min-h-screen items-center justify-center bg-gray-100 px-4">
      <div className="w-full max-w-sm rounded-xl bg-white p-6 shadow">
        <h1 className="mb-2 text-center text-2xl font-bold">
          MapMyStore
        </h1>
        <p className="mb-4 text-center text-gray-600">
          Retail Site Optimization System
        </p>

        {/* Error Message */}
        {error && (
          <div className="mb-4 rounded bg-red-100 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {/* Email */}
        <input
          type="email"
          placeholder="Email"
          className="mb-3 w-full rounded border p-2 focus:outline-none focus:ring"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />

        {/* Password */}
        <input
          type="password"
          placeholder="Password"
          className="mb-4 w-full rounded border p-2 focus:outline-none focus:ring"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />

        {/* Login Button */}
        <button
          onClick={handleLogin}
          disabled={loading}
          className={`w-full rounded py-2 text-white ${
            loading
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-blue-600 hover:bg-blue-700"
          }`}
        >
          {loading ? "Authenticating..." : "Login"}
        </button>

        {/* Demo Credentials */}
        <p className="mt-4 text-center text-xs text-gray-500">
          Demo login: admin@mapmystore.com / admin123
        </p>
      </div>
    </main>
  );
}
