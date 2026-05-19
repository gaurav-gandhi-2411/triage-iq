# Gemini 2.5 Flash — 20 RPD free-tier wall confirmed

**Date:** 2026-05-19  
**Model:** gemini-2.5-flash  
**Provider:** Google AI Studio (free tier)

## What happened

Attempted to use `gemini-2.5-flash` as a cross-family judge for 180 judge calls.
After 6 successful calls (5 from prior test runs + 1 in this session), every subsequent
call returned:

```
429 RESOURCE_EXHAUSTED
Quota exceeded for: generativelanguage.googleapis.com/generate_content_free_tier_requests
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20
```

## Conclusion

The free-tier daily limit for `gemini-2.5-flash` is **20 requests/day**, not the
1,500 RPD figure cited in third-party documentation (that figure applies to legacy
`gemini-1.5-flash`, which was deprecated and removed from the API on 2026-05-18).
`gemini-2.5-flash` on a new free-tier key has 20 RPD — confirmed by the API error.

No output file was written. Eval superseded by the Cohere Command A run.
