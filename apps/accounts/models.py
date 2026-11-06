from django.contrib.auth.models import AbstractUser
from django.db import models

from apps.common.constants import Role


class User(AbstractUser):
    role = models.CharField(max_length=16, choices=Role.choices, default=Role.VIEWER)

    @property
    def sees_all_documents(self) -> bool:
        return self.is_superuser or self.role == Role.ADMIN
