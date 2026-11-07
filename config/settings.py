"""Django settings. All deployment-specific values come from config.env (backed by .env)."""

from datetime import timedelta

from config.env import BASE_DIR, env, split_csv

SECRET_KEY = env.SECRET_KEY
DEBUG = env.DEBUG
ALLOWED_HOSTS = split_csv(env.ALLOWED_HOSTS)
CORS_ALLOWED_ORIGINS = split_csv(env.CORS_ALLOWED_ORIGINS)
CSRF_TRUSTED_ORIGINS = split_csv(env.CSRF_TRUSTED_ORIGINS)
CORS_ALLOW_CREDENTIALS = True

INSTALLED_APPS = [
    # Django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "corsheaders",
    "django_filters",
    "axes",
    "django_prometheus",
    # Local applications
    "apps.accounts",
    "apps.common",
    "apps.documents",
    "apps.retrieval",
    "apps.chatbot",
    "apps.evaluation",
    "apps.security",
    "apps.monitoring",
    "apps.reports",
]

MIDDLEWARE = [
    # First and last, so that request counts and latencies cover everything in between.
    "django_prometheus.middleware.PrometheusBeforeMiddleware",
    "apps.common.middleware.RequestIdMiddleware",
    "apps.common.middleware.RequestSizeMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    # After the views' middleware, so that it sees the outcome of each sign-in attempt.
    "axes.middleware.AxesMiddleware",
    "django_prometheus.middleware.PrometheusAfterMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        # The built React page wins over the "not built yet" placeholder in templates/.
        "DIRS": [BASE_DIR / "frontend" / "dist", BASE_DIR / "templates"],
        # Listed without Django's cached loader: a frontend build replaces index.html and deletes the files
        # the old copy points at, so a cached copy gives a blank page with 404s until the next restart.
        "OPTIONS": {
            "loaders": [
                "django.template.loaders.filesystem.Loader",
                "django.template.loaders.app_directories.Loader",
            ],
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": env.DATABASE_NAME,
        "USER": env.DATABASE_USER,
        "PASSWORD": env.DATABASE_PASSWORD,
        "HOST": env.DATABASE_HOST,
        "PORT": env.DATABASE_PORT,
        "CONN_MAX_AGE": 60,
    }
}

AUTH_USER_MODEL = "accounts.User"
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = False
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [p for p in (BASE_DIR / "static", BASE_DIR / "frontend" / "dist") if p.exists()]
WHITENOISE_USE_FINDERS = DEBUG
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": env.CACHE_URL,
        "KEY_PREFIX": "rageval",
    }
}

REST_FRAMEWORK = {
    # Sessions for the dashboard, JWTs for API clients that cannot keep a cookie. Both carry the same user, so the
    # same roles and document permissions apply.
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_FILTER_BACKENDS": ["django_filters.rest_framework.DjangoFilterBackend"],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.LimitOffsetPagination",
    "PAGE_SIZE": 50,
    "EXCEPTION_HANDLER": "apps.common.exceptions.api_exception_handler",
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    # Counted in Redis. "query" covers chat and search, which each hold the CPU for seconds; "user" is a ceiling for
    # everything else, high enough for the dashboard's polling.
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "login": "10/min",
        "anon": "60/min",
        "user": "600/min",
        "query": env.QUERY_RATE_LIMIT,
        "upload": "60/hour",
        # A Word or PDF file starts LibreOffice for a second or two.
        "download": "120/hour",
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=15),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=1),
    "AUTH_HEADER_TYPES": ("Bearer",),
}

# Sign-in lockout (django-axes). Counted per username and address together: all users of this machine share one
# address, so counting by address alone would lock everyone out after one person's typos.
AUTHENTICATION_BACKENDS = ["axes.backends.AxesStandaloneBackend", "django.contrib.auth.backends.ModelBackend"]
AXES_FAILURE_LIMIT = 5
AXES_COOLOFF_TIME = timedelta(minutes=15)
AXES_LOCKOUT_PARAMETERS = [["username", "ip_address"]]
AXES_RESET_ON_SUCCESS = True
AXES_LOCKOUT_CALLABLE = "apps.accounts.lockout.lockout_response"

# Celery. A single worker with the solo pool keeps one copy of the ML models in RAM.
CELERY_BROKER_URL = env.CELERY_BROKER_URL
CELERY_RESULT_BACKEND = env.CELERY_RESULT_BACKEND
# Tasks are acknowledged on receipt. With late acks, Redis redelivers any task still running after its
# one-hour visibility timeout, and long evaluation runs would then execute twice. A run interrupted by a
# crash is resumed by apps.evaluation.tasks instead.
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_DEFAULT_QUEUE = "rageval"
CELERY_TIMEZONE = TIME_ZONE

# Documents
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_BYTES
# JSON requests: a question, a dataset of cases. Larger is a mistake or an attack, and gets a 413.
MAX_JSON_BODY_BYTES = 10 * 1024 * 1024
# Signed-in sessions end after 12 hours without use; each request moves the end time on.
SESSION_COOKIE_AGE = 12 * 3600
SESSION_SAVE_EVERY_REQUEST = True
ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"}

NLI_MODEL = env.NLI_MODEL

# Targets for live chat over the chosen period. Cache hit rate is shown beside them, without a target: it depends on
# what people ask more than on the system.
SLO = {
    "p95_ms": env.SLO_P95_MS,
    "first_token_p95_ms": env.SLO_FIRST_TOKEN_P95_MS,
    "error_rate": env.SLO_ERROR_RATE,
    "tokens_per_second": env.SLO_MIN_TOKENS_PER_SECOND,
    "answer_rate": env.SLO_MIN_ANSWER_RATE,
    "cost_per_query": env.SLO_MAX_COST_PER_QUERY,
}

# Evaluation thresholds. They live here so they are captured in each run's snapshot.
EVALUATION = {
    # Fixed for all runs: if it followed each RagConfig, scores across configs could not be compared.
    "JUDGE_EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    # Probability above which the NLI model is believed.
    "ENTAILMENT_THRESHOLD": 0.5,
    "CONTRADICTION_THRESHOLD": 0.7,
    # Cosine similarity above which a context sentence counts as relevant to the question.
    "CONTEXT_RELEVANCY_THRESHOLD": 0.35,
    # Share of an evidence string's tokens that must appear in a chunk for the chunk to count as gold.
    "EVIDENCE_MATCH_THRESHOLD": 0.8,
}

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {"request_id": {"()": "apps.common.middleware.RequestIdLogFilter"}},
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "plain", "filters": ["request_id"]}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {
        "LiteLLM": {"level": "WARNING"},
        "httpx": {"level": "WARNING"},
        # Otherwise it logs one warning per recognizer for languages this project does not use.
        "presidio-analyzer": {"level": "ERROR"},
        "sentence_transformers": {"level": "WARNING"},
    },
}

if not DEBUG:
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_CONTENT_TYPE_NOSNIFF = True
