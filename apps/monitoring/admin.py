from django.contrib import admin

from apps.monitoring.models import CostRecord, PerformanceLog, TokenUsage

for model in (PerformanceLog, TokenUsage, CostRecord):
    admin.site.register(model)
