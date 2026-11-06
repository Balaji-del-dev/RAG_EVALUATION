from rest_framework.permissions import SAFE_METHODS, BasePermission

from apps.common.constants import Role


def has_role(user, *roles) -> bool:
    return bool(user and user.is_authenticated and (user.is_superuser or user.role in roles))


class IsAdminRole(BasePermission):
    def has_permission(self, request, view):
        return has_role(request.user, Role.ADMIN)


class IsAnalystOrAdmin(BasePermission):
    """For data that holds document text. Evaluation runs read the whole corpus, restricted documents
    included, so their answers and contexts must not be visible to viewers."""

    def has_permission(self, request, view):
        return has_role(request.user, Role.ADMIN, Role.ANALYST)


class IsAnalystOrReadOnly(BasePermission):
    """Viewers can read. Analysts and admins can change things and start runs."""

    def has_permission(self, request, view):
        if not (request.user and request.user.is_authenticated):
            return False
        return request.method in SAFE_METHODS or has_role(request.user, Role.ADMIN, Role.ANALYST)
