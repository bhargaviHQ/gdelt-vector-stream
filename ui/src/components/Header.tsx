import { useEffect, useState } from "react";
import { fetchHealth, fetchStats, triggerIngest } from "../api";
import type { HealthResponse, StatsResponse } from "../types";

const DOT_COLOR: Record<string, string> = {
  ok: "bg-green-500",
  degraded: "bg-yellow-500",
  error: "bg-red-500",
};

export default function Header() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [ingesting, setIngesting] = useState(false);

  const poll = () => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
    fetchStats().then(setStats).catch(() => setStats(null));
  };

  useEffect(() => {
    poll();
    const id = setInterval(poll, 30_000);
    return () => clearInterval(id);
  }, []);

  const handleIngest = async () => {
    setIngesting(true);
    try {
      await triggerIngest(100, 1);
      poll();
    } catch (e) {
      alert(e instanceof Error ? e.message : "Ingestion failed");
    } finally {
      setIngesting(false);
    }
  };

  return (
    <header className="flex items-center justify-between border-b border-gray-800 px-6 py-4">
      <h1 className="text-lg font-semibold tracking-tight text-gray-100">
        GDELT Vector Stream
      </h1>

      <div className="flex items-center gap-4">
        {/* Stats badges */}
        {stats && (
          <>
            <span className="rounded-md bg-gray-800 px-3 py-1 text-xs text-gray-300">
              {stats.vector_count.toLocaleString()} vectors
            </span>
            <span className="rounded-md bg-gray-800 px-3 py-1 text-xs text-gray-300">
              {stats.processed_files} files
            </span>
          </>
        )}

        {/* Health dot */}
        <span
          className={`h-2.5 w-2.5 rounded-full ${health ? DOT_COLOR[health.status] : "bg-gray-600"}`}
          title={health ? `Status: ${health.status}` : "Connecting..."}
        />

        {/* Ingest button */}
        <button
          onClick={handleIngest}
          disabled={ingesting}
          className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {ingesting ? "Ingesting..." : "Ingest Data"}
        </button>
      </div>
    </header>
  );
}
