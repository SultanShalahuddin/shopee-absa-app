create extension if not exists "pgcrypto";

create table if not exists public.reviews (
  id uuid primary key default gen_random_uuid(),
  review_text text not null check (char_length(review_text) between 3 and 2000),
  model_version text not null,
  latency_ms integer not null check (latency_ms >= 0),
  created_at timestamptz not null default now()
);

create table if not exists public.aspect_predictions (
  id uuid primary key default gen_random_uuid(),
  review_id uuid not null references public.reviews(id) on delete cascade,
  aspect text not null,
  aspect_label text not null,
  sentiment text not null check (sentiment in ('tidak_disebut', 'negatif', 'netral', 'positif')),
  confidence double precision not null check (confidence between 0 and 1),
  created_at timestamptz not null default now(),
  unique (review_id, aspect)
);

create index if not exists reviews_created_at_idx on public.reviews(created_at desc);
create index if not exists predictions_review_id_idx on public.aspect_predictions(review_id);
create index if not exists predictions_aspect_sentiment_idx on public.aspect_predictions(aspect, sentiment);

alter table public.reviews enable row level security;
alter table public.aspect_predictions enable row level security;

-- Aplikasi menulis dan membaca melalui FastAPI menggunakan service role key.
-- Service role melewati RLS. Jangan pernah menaruh service role key di frontend.
