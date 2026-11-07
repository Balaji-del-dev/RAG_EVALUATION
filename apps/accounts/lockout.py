from django.conf import settings
from django.http import JsonResponse


def lockout_response(request, original_response=None, credentials=None):
    """JSON, so that the sign-in screen can show it like any other error."""
    minutes = int(settings.AXES_COOLOFF_TIME.total_seconds() // 60)
    return JsonResponse({"error": f"Too many failed sign-ins. Try again in {minutes} minutes."}, status=429)
