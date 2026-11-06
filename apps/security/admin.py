from django.contrib import admin

from apps.security.models import AuditLog, SecurityResult, SecurityTest


@admin.register(SecurityTest)
class SecurityTestAdmin(admin.ModelAdmin):
    list_display = ["test_name", "attack_type", "expected_block", "severity", "enabled"]
    list_filter = ["attack_type", "severity", "enabled"]


@admin.register(SecurityResult)
class SecurityResultAdmin(admin.ModelAdmin):
    list_display = ["run", "test", "passed", "guard_blocked", "attack_succeeded", "severity"]
    list_filter = ["passed", "severity"]


@admin.register(AuditLog)
class AuditLogAdmin(admin.ModelAdmin):
    list_display = ["timestamp", "user", "action", "target"]
    list_filter = ["action"]

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False
