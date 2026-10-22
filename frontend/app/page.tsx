"use client";
import { useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
const IMG = "https://image.tmdb.org/t/p/w500";
const CHIPS = ["Feel-good comedies under 100 minutes", "Mind-bending thrillers like Inception", "Animated movies for the whole family"];

type Movie = { id: number; title: string; year: number | null; runtime: number; genres: string[]; overview: string; poster_path: string; reason?: string | null };
type Applied = { genres: string[]; max_runtime: number | null; min_year: number | null; max_year: number | null; similar_to: string | null };

const rt = (m: number) => (m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m`);
const grid = "grid grid-cols-2 gap-4 min-[480px]:grid-cols-3 md:grid-cols-4 min-[1100px]:grid-cols-5";
const pill = "rounded-full border border-[var(--border)] bg-[var(--surface-2)] px-4 py-2 text-sm";
const btn = "min-h-11 rounded-full bg-[var(--accent)] px-6 font-semibold text-black hover:bg-[var(--accent-hover)] disabled:opacity-40";

function Poster({ m }: { m: Movie }) {
  return m.poster_path ? (
    // eslint-disable-next-line @next/next/no-img-element
    <img src={IMG + m.poster_path} alt={m.title} loading="lazy" className="aspect-[2/3] w-full rounded-xl object-cover shadow-lg" />
  ) : (
    <div className="flex aspect-[2/3] w-full items-center justify-center rounded-xl bg-[var(--surface-2)] p-2 text-center text-sm">{m.title}</div>
  );
}

function Stars({ value, onChange }: { value: number; onChange: (n: number) => void }) {
  return (
    <div className="flex">
      {[1, 2, 3, 4, 5].map((n) => (
        <button key={n} aria-label={`${n} star${n > 1 ? "s" : ""}`} onClick={() => onChange(n)} className="h-11 w-11 text-3xl" style={{ color: n <= value ? "var(--accent)" : "var(--border)" }}>★</button>
      ))}
    </div>
  );
}

function Sheet({ m, rating, onRate, onClose }: { m: Movie; rating: number; onRate: (n: number) => void; onClose: () => void }) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 md:items-center" onClick={onClose}>
      <div role="dialog" aria-modal="true" aria-label={m.title} onClick={(e) => e.stopPropagation()} className="fade max-h-[90vh] w-full overflow-y-auto rounded-t-2xl bg-[var(--surface)] p-6 md:max-w-2xl md:rounded-2xl">
        <button onClick={onClose} aria-label="Close" className="float-right h-11 w-11 text-xl">✕</button>
        <div className="flex flex-col gap-6 sm:flex-row">
          <div className="w-40 shrink-0 self-center sm:self-start"><Poster m={m} /></div>
          <div>
            <h2 className="text-2xl font-semibold">{m.title}</h2>
            <p className="mt-1 text-sm font-medium text-[var(--text-muted)]">{m.year} · {rt(m.runtime)}</p>
            <div className="mt-3 flex flex-wrap gap-2">{m.genres.map((g) => <span key={g} className="rounded-full bg-[var(--surface-2)] px-3 py-1 text-xs">{g}</span>)}</div>
            <p className="mt-4 text-[var(--text-muted)]">{m.overview}</p>
            <p className="mt-4 text-sm font-medium">Rate this movie</p>
            <Stars value={rating} onChange={onRate} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  const [stage, setStage] = useState<"landing" | "onboarding" | "main">("landing");
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [intro, setIntro] = useState<Movie[]>([]);
  const [results, setResults] = useState<Movie[] | null>(null);
  const [applied, setApplied] = useState<Applied | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Movie | null>(null);
  const [query, setQuery] = useState("");
  const last = useRef<string | null>(null);
  const count = Object.keys(ratings).length;

  async function call(fn: () => Promise<void>) {
    setLoading(true); setError(null); setSlow(false);
    const t = setTimeout(() => setSlow(true), 5000);
    try { await fn(); } catch { setError("We couldn't reach the server. Please try again."); }
    finally { clearTimeout(t); setLoading(false); setSlow(false); }
  }

  const run = (message: string | null, rs = ratings) => {
    last.current = message;
    return call(async () => {
      const payload = { ratings: Object.entries(rs).map(([id, rating]) => ({ movie_id: Number(id), rating })), limit: 20 };
      const res = await fetch(`${API}${message ? "/chat" : "/recommend"}`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(message ? { ...payload, message } : payload),
      });
      if (!res.ok) throw new Error();
      const d = await res.json();
      setResults(d.results); setApplied(d.filters); setNotice(d.notice);
    });
  };

  const startOnboarding = () => {
    setStage("onboarding");
    if (!intro.length) call(async () => {
      const res = await fetch(`${API}/onboarding?count=12`);
      if (!res.ok) throw new Error();
      setIntro((await res.json()).movies);
    });
  };
  const enterMain = () => { setStage("main"); run(null); };
  const rate = (m: Movie, n: number) => {
    const next = { ...ratings, [m.id]: n };
    setRatings(next);
    if (stage === "main") run(last.current, next);
  };
  const ask = (q: string) => { if (q.trim()) { setQuery(q); run(q); } };

  const tags: string[] = [];
  if (applied) {
    tags.push(...applied.genres);
    if (applied.max_runtime) tags.push(`Under ${applied.max_runtime % 60 === 0 ? applied.max_runtime / 60 + "h" : applied.max_runtime + " min"}`);
    if (applied.min_year) tags.push(`From ${applied.min_year}`);
    if (applied.max_year) tags.push(`Until ${applied.max_year}`);
    if (applied.similar_to) tags.push(`Like: ${applied.similar_to}`);
  }

  const errorBox = error && (
    <div role="alert" className="my-6 rounded-xl border border-[var(--danger)] p-4 text-[var(--danger)]">
      {error} <button className="ml-3 underline" onClick={() => (stage === "onboarding" ? startOnboarding() : run(last.current))}>Try again</button>
    </div>
  );
  const warming = loading && slow && <p className="my-4 text-center text-[var(--text-muted)]">Warming up the server, one moment…</p>;
  const skeletons = <div className={grid}>{Array.from({ length: 10 }).map((_, i) => <div key={i} className="shimmer aspect-[2/3] rounded-xl" />)}</div>;

  return (
    <div className="mx-auto flex min-h-screen max-w-[1200px] flex-col px-4 py-8">
      <main className="flex-1">
        {stage === "landing" && (
          <section className="fade flex min-h-[70vh] flex-col items-center justify-center text-center">
            <h1 className="wordmark text-7xl sm:text-8xl">REEL</h1>
            <p className="mt-4 max-w-md text-lg text-[var(--text-muted)]">Tell us what you love. We&apos;ll find what&apos;s next.</p>
            <button className={`${btn} mt-8`} onClick={startOnboarding}>Get started</button>
          </section>
        )}

        {stage === "onboarding" && (
          <section className="fade">
            <h2 className="text-3xl font-semibold">Rate a few movies you know</h2>
            <p className="mt-1 text-[var(--text-muted)]">Tap a poster to rate it. Five ratings tune your picks.</p>
            {errorBox}{warming}
            {loading && !intro.length ? <div className="mt-6">{skeletons}</div> : (
              <div className={`${grid} mt-6`}>
                {intro.map((m) => (
                  <button key={m.id} onClick={() => setSelected(m)} className="relative text-left">
                    <Poster m={m} />
                    {ratings[m.id] && <span className="absolute right-2 top-2 rounded-full bg-black/70 px-2 py-1 text-sm text-[var(--accent)]">{ratings[m.id]} ★</span>}
                  </button>
                ))}
              </div>
            )}
            <div className="sticky bottom-0 mt-6 flex items-center justify-between gap-4 bg-[var(--bg)]/90 py-4 backdrop-blur">
              <span className="text-sm text-[var(--text-muted)]">{Math.min(count, 5)} of 5 rated</span>
              <div className="flex items-center gap-4">
                <button className="min-h-11 text-sm text-[var(--text-muted)] underline" onClick={enterMain}>Skip</button>
                <button className={btn} disabled={count < 5} onClick={enterMain}>Continue</button>
              </div>
            </div>
          </section>
        )}

        {stage === "main" && (
          <section className="fade">
            <form onSubmit={(e) => { e.preventDefault(); ask(query); }} className="flex items-center gap-2 rounded-full border border-[var(--border)] bg-[var(--surface)] p-2 pl-6">
              <input value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Describe what you want to watch" placeholder="Try: sci-fi like Interstellar under 2 hours" className="min-h-11 flex-1 bg-transparent outline-none placeholder:text-[var(--text-muted)]" />
              <button type="submit" className={btn} disabled={loading}>Send</button>
            </form>
            <div className="mt-4 flex flex-wrap gap-2">{CHIPS.map((c) => <button key={c} className={pill} onClick={() => ask(c)}>{c}</button>)}</div>
            {tags.length > 0 && <div className="mt-4 flex flex-wrap gap-2">{tags.map((t) => <span key={t} className="rounded-full border border-[var(--accent)] px-3 py-1 text-sm text-[var(--accent)]">{t}</span>)}</div>}
            {notice && <p className="mt-4 text-sm text-[var(--text-muted)]">{notice}</p>}
            {errorBox}{warming}
            <div className="mt-6">
              {loading ? skeletons : results && results.length === 0 ? (
                <div className="py-12 text-center">
                  <p className="text-[var(--text-muted)]">No matches for those filters. Try loosening the runtime or genre.</p>
                  <div className="mt-4 flex flex-wrap justify-center gap-2">{CHIPS.map((c) => <button key={c} className={pill} onClick={() => ask(c)}>{c}</button>)}</div>
                </div>
              ) : (
                <div className={grid}>
                  {results?.map((m, i) => (
                    <button key={m.id} onClick={() => setSelected(m)} style={{ animationDelay: `${i * 40}ms` }} className="fade group text-left">
                      <div className="rounded-xl border border-transparent transition duration-150 group-hover:scale-[1.03] group-hover:border-[var(--accent)] group-hover:shadow-[0_0_24px_rgba(245,184,65,0.25)]"><Poster m={m} /></div>
                      <p className="mt-2 truncate font-semibold">{m.title}</p>
                      <p className="text-[13px] font-medium text-[var(--text-muted)]">{m.year} · {rt(m.runtime)} · {m.genres[0]}</p>
                      {m.reason && <p className="mt-1 text-xs text-[var(--text-muted)]">{m.reason}</p>}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      <footer className="mt-12 border-t border-[var(--border)] pt-6 text-xs text-[var(--text-muted)]">
        <p>This product uses the TMDB API but is not endorsed or certified by TMDB.</p>
        <p className="mt-1">Built by Navaneeth M · Python for AI</p>
      </footer>

      {selected && <Sheet m={selected} rating={ratings[selected.id] ?? 0} onRate={(n) => rate(selected, n)} onClose={() => setSelected(null)} />}
    </div>
  );
}
