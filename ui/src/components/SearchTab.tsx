import { useState } from "react";
import { searchEvents } from "../api";
import type { SearchResult } from "../types";
import EventCard from "./EventCard";

export default function SearchTab() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(5);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searched, setSearched] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setSearched(true);
    try {
      const res = await searchEvents(query.trim(), topK);
      setResults(res.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleSearch} className="flex gap-3">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search GDELT events..."
          className="flex-1 rounded-lg border border-gray-700 bg-gray-800 px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 outline-none focus:border-blue-500"
        />
        <select
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="rounded-lg border border-gray-700 bg-gray-800 px-3 py-2.5 text-sm text-gray-300 outline-none"
        >
          {[3, 5, 10, 15, 20].map((n) => (
            <option key={n} value={n}>
              Top {n}
            </option>
          ))}
        </select>
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading && (
        <p className="text-center text-sm text-gray-400 animate-pulse">
          Searching...
        </p>
      )}

      {!loading && searched && results.length === 0 && !error && (
        <p className="text-center text-sm italic text-gray-500">
          No results found. Try a different query.
        </p>
      )}

      {results.length > 0 && (
        <div className="space-y-3">
          <p className="text-xs text-gray-400">
            {results.length} result{results.length !== 1 ? "s" : ""} for "
            {query}"
          </p>
          {results.map((r) => (
            <EventCard key={r.vector_id} result={r} />
          ))}
        </div>
      )}
    </div>
  );
}
