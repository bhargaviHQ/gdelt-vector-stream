import { useState } from "react";
import { askQuestion } from "../api";
import type { AskResponse } from "../types";
import EventCard from "./EventCard";

export default function AskTab() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSources, setShowSources] = useState(false);

  const handleAsk = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResponse(null);
    setShowSources(false);
    try {
      const res = await askQuestion(question.trim());
      setResponse(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <form onSubmit={handleAsk} className="flex gap-3">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about global events..."
          className="flex-1 rounded-lg border border-gray-700 bg-gray-800 px-4 py-2.5 text-sm text-gray-100 placeholder-gray-500 outline-none focus:border-blue-500"
        />
        <button
          type="submit"
          disabled={loading || !question.trim()}
          className="rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {loading ? "Thinking..." : "Ask"}
        </button>
      </form>

      {error && (
        <div className="rounded-lg border border-red-700 bg-red-900/30 p-3 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading && (
        <p className="text-center text-sm text-gray-400 animate-pulse">
          Thinking...
        </p>
      )}

      {response?.error && (
        <div className="rounded-lg border border-yellow-700 bg-yellow-900/30 p-3 text-sm text-yellow-300">
          {response.message || "Something went wrong with the LLM."}
        </div>
      )}

      {response?.answer && (
        <div className="rounded-xl border border-gray-700 bg-gray-800 p-6 space-y-4">
          <p className="text-sm text-gray-100 whitespace-pre-wrap leading-relaxed">
            {response.answer}
          </p>
          <p className="text-xs text-gray-500">
            Answered via {response.model}
          </p>
        </div>
      )}

      {response && response.events.length > 0 && (
        <div className="space-y-3">
          <button
            onClick={() => setShowSources(!showSources)}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            {showSources ? "Hide" : "Show"} {response.events.length} source
            event{response.events.length !== 1 ? "s" : ""}
          </button>
          {showSources &&
            response.events.map((r) => (
              <EventCard key={r.vector_id} result={r} />
            ))}
        </div>
      )}
    </div>
  );
}
