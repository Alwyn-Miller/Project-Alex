export interface Result<T = unknown> {
  data?: T | null;
  error?: string | null;
  success: boolean;
}
