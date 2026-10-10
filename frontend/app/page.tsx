"use client";

import { useEffect, useMemo, useState } from "react";
import { BarChart3, Clock3, Github, LoaderCircle, Moon, Sparkles, Sun, WandSparkles } from "lucide-react";
import HistoryList from "@/components/HistoryList";
import SentimentCard from "@/components/SentimentCard";
import SentimentSummary from "@/components/SentimentSummary";
import { fetchHistory, HistoryItem, predictReview, PredictionResponse } from "@/lib/api";

const EXAMPLE = "Aplikasinya cepat dan mudah digunakan, tetapi kurirnya lambat dan voucher tidak bisa dipakai.";

export default function Home() {
  const [review, setReview] = useState("");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showAll, setShowAll] = useState(true);
  const [dark, setDark] = useState(false);

  useEffect(() => { fetchHistory().then(setHistory); }, []);
  useEffect(() => { document.documentElement.dataset.theme = dark ? "dark" : "light"; }, [dark]);

  const visiblePredictions = useMemo(() => {
    if (!result) return [];
    return showAll ? result.predictions : result.predictions.filter((x) => x.sentiment !== "tidak_disebut");
  }, [result, showAll]);

  async function submit() {
    if (review.trim().length < 3) { setError("Masukkan review minimal 3 karakter."); return; }
    setLoading(true); setError("");
    try {
      const prediction = await predictReview(review.trim());
      setResult(prediction);
      setHistory(await fetchHistory());
      document.getElementById("hasil")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Terjadi kesalahan.");
    } finally { setLoading(false); }
  }

  return (
    <main>
      <nav className="nav">
        <a className="brand" href="#"><span><BarChart3 size={21} /></span>ReviewScope</a>
        <div className="nav-links"><a href="#analisis">Analisis</a><a href="#hasil">Hasil</a><a href="#riwayat">Riwayat</a></div>
        <div className="nav-actions">
          <button className="icon-button" onClick={() => setDark((v) => !v)} aria-label="Ubah tema">{dark ? <Sun /> : <Moon />}</button>
          <a className="icon-button" href="https://github.com" aria-label="GitHub"><Github /></a>
        </div>
      </nav>

      <section className="hero" id="analisis">
        <div className="blob blob-left" /><div className="blob blob-right" />
        <div className="hero-copy">
          <span className="hero-badge"><Sparkles size={15} /> IndoBERT Aspect-Based Sentiment Analysis</span>
          <h1>Pahami suara pelanggan dengan lebih jelas.</h1>
          <p>Analisis satu review menjadi sentimen untuk delapan aspek layanan Shopee secara otomatis.</p>
        </div>
        <div className="input-card">
          <div className="input-top"><label htmlFor="review">Masukkan ulasan pelanggan</label><span>{review.length}/2000</span></div>
          <textarea id="review" value={review} maxLength={2000} onChange={(e) => setReview(e.target.value)} placeholder="Contoh: aplikasinya bagus, tetapi pengiriman lambat..." />
          <div className="input-footer">
            <button className="example-button" onClick={() => setReview(EXAMPLE)}>Gunakan contoh</button>
            <button className="primary-button" onClick={submit} disabled={loading}>
              {loading ? <LoaderCircle className="spin" size={18} /> : <WandSparkles size={18} />}
              {loading ? "Menganalisis..." : "Analisis Sentimen"}
            </button>
          </div>
          {error && <p className="error-message">{error}</p>}
        </div>
        <p className="privacy-note">Review diproses oleh IndoBERT dan disimpan hanya untuk riwayat analisis aplikasi.</p>
      </section>

      <section className="results-section" id="hasil">
        <div className="section-title-row">
          <div><span className="eyebrow">Hasil model</span><h2>Analisis berdasarkan aspek</h2><p>Setiap aspek dapat memiliki sentimen berbeda dalam review yang sama.</p></div>
          {result && <label className="toggle"><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} /><span />Tampilkan semua aspek</label>}
        </div>

        {!result ? (
          <div className="empty-result"><WandSparkles size={30} /><h3>Hasil analisis akan muncul di sini</h3><p>Masukkan review di atas untuk mulai menggunakan model.</p></div>
        ) : (
          <>
            {result.demo_mode && <div className="demo-banner">Mode demo aktif. Pasang folder model IndoBERT untuk menggunakan prediksi asli.</div>}
            <div className="results-layout">
              <div className="aspect-grid">{visiblePredictions.map((p) => <SentimentCard key={p.aspect} prediction={p} />)}</div>
              <SentimentSummary predictions={result.predictions} />
            </div>
            <div className="model-meta"><span><Clock3 size={15} /> {result.latency_ms} ms</span><span>Model: {result.model_version}</span><span>{result.stored ? "Tersimpan di database" : "Belum tersimpan"}</span></div>
          </>
        )}
      </section>
      <HistoryList items={history} />
      <footer><span>ReviewScope ABSA</span><span>IndoBERT • FastAPI • Supabase</span></footer>
    </main>
  );
}
