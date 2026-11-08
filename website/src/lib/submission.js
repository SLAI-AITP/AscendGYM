export function validateSubmitter(input = {}) {
  const value = Object.fromEntries(['organization', 'email', 'name'].map(key => [key, String(input[key] ?? '').trim()]));
  if (!value.organization || !value.email || !value.name) {
    return { ok: false, message: 'Please complete your organization, email address, and full name before selecting files.' };
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.email)) {
    return { ok: false, message: 'Please enter a valid email address before selecting files.' };
  }
  return { ok: true, value };
}
