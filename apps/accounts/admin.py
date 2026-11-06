from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from apps.accounts.models import User


@admin.register(User)
class AppUserAdmin(UserAdmin):
    list_display = ["username", "email", "role", "is_active", "is_superuser"]
    list_filter = ["role", "is_active"]
    fieldsets = [*UserAdmin.fieldsets, ("Platform", {"fields": ["role"]})]
