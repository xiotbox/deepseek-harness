import { describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import AgentRegistry, { type Agent } from '@deepseek-ai/dsh-agent'
import UserQuestionService, {
  UserQuestionError,
  type AskUserQuestionAnswer,
  type AskUserQuestionRequest,
} from '@deepseek-ai/dsh-user-questions'

interface QuestionAnswerer {
  ask(request: AskUserQuestionRequest): Promise<AskUserQuestionAnswer>
}

function registerAnswerer(ctx: Context, answerer: QuestionAnswerer): () => void {
  return ctx.on('user-questions/request', request => answerer.ask(request))
}

function provider(answer = 'approved'): QuestionAnswerer & { seen: AskUserQuestionRequest[] } {
  const seen: AskUserQuestionRequest[] = []
  return {
    seen,
    async ask(request) {
      seen.push(request)
      return {
        answers: request.questions.map(question => ({ id: question.id, selected: [answer] })),
      }
    },
  }
}

function stubAgent(id: string, delegationDepth = 0): Agent {
  const agentId = id as Agent['id']
  return {
    id: agentId,
    session: { id: agentId, header: { delegationDepth } },
  } as unknown as Agent
}

describe('UserQuestionService', () => {
  it('delegates ask requests to the registered provider', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = provider('yes')
    registerAnswerer(ctx, p)
    const questions = [{ id: 'confirm', question: 'Proceed?', options: [{ label: 'yes' }] }]

    const result = await ctx.userQuestions.ask({ questions })

    expect(result).toEqual({ answers: [{ id: 'confirm', selected: ['yes'] }] })
    expect(p.seen).toEqual([{ questions, signal: expect.any(AbortSignal) }])
  })

  it('rejects ask requests when no provider is registered', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)

    await expect(ctx.userQuestions.ask({ questions: [{ id: 'confirm', question: 'Proceed?' }] }))
      .rejects.toMatchObject({ name: 'UserQuestionError', code: 'NO_PROVIDER' })
  })

  it('registers providers with HMR-safe disposal', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = provider()
    const dispose = registerAnswerer(ctx, p)

    dispose()
    dispose()

    await expect(ctx.userQuestions.ask({ questions: [{ id: 'confirm', question: 'Proceed?' }] }))
      .rejects.toMatchObject({ code: 'NO_PROVIDER' })
  })

  it('delegates through composed answerers', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const delegated = vi.fn()
    ctx.on('user-questions/request', (_request, next) => {
      delegated()
      return next()
    })
    const p = provider('second')
    registerAnswerer(ctx, p)

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?', options: [{ label: 'second' }] }],
    })).resolves.toEqual({ answers: [{ id: 'confirm', selected: ['second'] }] })
    expect(delegated).toHaveBeenCalledOnce()
  })

  it('fails before reaching the provider when the signal is already aborted', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [{ id: 'confirm', selected: ['too late'] }] })) }
    registerAnswerer(ctx, p)
    const controller = new AbortController()
    controller.abort()

    await expect(ctx.userQuestions.ask({ questions: [{ id: 'confirm', question: 'Proceed?' }], signal: controller.signal }))
      .rejects.toMatchObject({ code: 'ASK_ABORTED' })
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('normalizes an in-flight signal cancellation to ASK_ABORTED', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const pending = Promise.withResolvers<never>()
    registerAnswerer(ctx, { ask: () => pending.promise })
    const controller = new AbortController()
    const abortReason = new DOMException('This operation was aborted', 'AbortError')

    const answer = ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
      signal: controller.signal,
    })
    controller.abort(abortReason)
    pending.reject(abortReason)

    await expect(answer).rejects.toMatchObject({
      name: 'UserQuestionError',
      code: 'ASK_ABORTED',
      cause: abortReason,
    })
  })

  it('preserves a domain rejection when its provider also aborts the signal', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const controller = new AbortController()
    const cancelled = new UserQuestionError('the user cancelled ask_user_question', 'ASK_CANCELLED')
    registerAnswerer(ctx, {
      ask: () => {
        controller.abort()
        return Promise.reject(cancelled)
      },
    })

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
      signal: controller.signal,
    })).rejects.toBe(cancelled)
  })

  it('restores a transported provider rejection to UserQuestionError', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const transported = Object.assign(new Error('the user cancelled ask_user_question'), {
      name: 'UserQuestionError',
      code: 'ASK_CANCELLED',
    })
    registerAnswerer(ctx, { ask: () => Promise.reject(transported) })

    const rejection = await ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
    }).then(
      () => undefined,
      (error: unknown) => error,
    )

    expect(rejection).toBeInstanceOf(UserQuestionError)
    expect(rejection).toMatchObject({
      name: 'UserQuestionError',
      code: 'ASK_CANCELLED',
      cause: transported,
    })
  })

  it.each([
    ['an ordinary Error', new Error('provider failed')],
    ['a namesake Error without a string code', Object.assign(new Error('provider failed'), {
      name: 'UserQuestionError',
    })],
    ['a non-Error rejection', { name: 'UserQuestionError', code: 'ASK_CANCELLED' }],
  ])('preserves %s from the provider', async (_label, rejection) => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    registerAnswerer(ctx, { ask: vi.fn().mockRejectedValue(rejection) })

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
    })).rejects.toBe(rejection)
  })

  it('rejects empty question batches before reaching the provider', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [] })) }
    registerAnswerer(ctx, p)

    await expect(ctx.userQuestions.ask({ questions: [] }))
      .rejects.toMatchObject({ name: 'UserQuestionError', code: 'EMPTY_QUESTIONS' })
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('rejects a live runtime-owned agent before reaching the provider', async () => {
    const ctx = new Context()
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [] })) }
    registerAnswerer(ctx, p)
    const root = stubAgent('root', 0)
    const child = stubAgent('child', 0)
    ctx.agents.enter(root, undefined)
    ctx.agents.enter(child, root)

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
      agent: child,
    })).rejects.toMatchObject({
      name: 'UserQuestionError',
      code: 'DELEGATED_CALLER',
      message: "human interaction is unavailable while the calling agent is owned by another live agent; include the unresolved question or decision in the child agent's final result",
    })
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('reaches the provider for a lineage-bearing session resumed as a runtime root', async () => {
    const ctx = new Context()
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(UserQuestionService)
    const p = provider('yes')
    registerAnswerer(ctx, p)
    const agent = stubAgent('resumed-root', 1)
    ctx.agents.enter(agent, undefined)

    const result = await ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?', options: [{ label: 'yes' }] }],
      agent,
    })

    expect(result).toEqual({ answers: [{ id: 'confirm', selected: ['yes'] }] })
  })

  it('rejects a supplied agent when no live registry can attest it', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [] })) }
    registerAnswerer(ctx, p)

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
      agent: stubAgent('unattested'),
    })).rejects.toMatchObject({ name: 'UserQuestionError', code: 'CALLER_NOT_LIVE' })
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('rejects a stale agent object that reuses a live id', async () => {
    const ctx = new Context()
    await ctx.plugin(AgentRegistry)
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [] })) }
    registerAnswerer(ctx, p)
    const live = stubAgent('same-id')
    ctx.agents.enter(live, undefined)

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
      agent: stubAgent('same-id'),
    })).rejects.toMatchObject({ name: 'UserQuestionError', code: 'CALLER_NOT_LIVE' })
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('restores a transported UserQuestionError to the public error class', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const transported = Object.assign(new Error('the user cancelled ask_user_question'), {
      name: 'UserQuestionError',
      code: 'ASK_CANCELLED',
    })
    registerAnswerer(ctx, { ask: () => Promise.reject(transported) })

    const failure = await ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
    }).then(() => undefined, (error: unknown) => error)

    expect(failure).toBeInstanceOf(UserQuestionError)
    expect(failure).toMatchObject({
      name: 'UserQuestionError', code: 'ASK_CANCELLED', cause: transported,
    })
  })

  it('preserves a provider rejection outside the UserQuestionError taxonomy', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const failure = new Error('provider failed')
    registerAnswerer(ctx, { ask: () => Promise.reject(failure) })

    await expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
    })).rejects.toBe(failure)
  })

  it('rejects an intent whose approve label names none of its own options', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [] })) }
    registerAnswerer(ctx, p)
    const question = { id: 'plan-review', question: 'Approve?', detail: '# Plan' }

    // A wrong label among offered options, and no options offered at all.
    for (const options of [[{ label: 'Approve' }], undefined]) {
      await expect(ctx.userQuestions.ask({
        questions: [{
          ...question,
          ...(options === undefined ? {} : { options }),
          intent: { kind: 'plan-review', approve: 'Ship it' },
        }],
      })).rejects.toMatchObject({ name: 'UserQuestionError', code: 'BAD_INTENT' })
    }
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('rejects a plan-review intent on a question carrying no plan to review', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = { ask: vi.fn(async () => ({ answers: [] })) }
    registerAnswerer(ctx, p)

    // Detail IS the plan for this intent, so a UI honouring it would ask the
    // user to approve something they cannot see.
    await expect(ctx.userQuestions.ask({
      questions: [{
        id: 'plan-review', question: 'Approve?',
        options: [{ label: 'Approve' }, { label: 'Keep planning' }],
        intent: { kind: 'plan-review', approve: 'Approve' },
      }],
    })).rejects.toMatchObject({ name: 'UserQuestionError', code: 'BAD_INTENT' })
    expect(p.ask).not.toHaveBeenCalled()
  })

  it('passes an intent through once its approve label names an offered option', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const p = provider('Approve')
    registerAnswerer(ctx, p)
    const intent = { kind: 'plan-review', approve: 'Approve' } as const

    const result = await ctx.userQuestions.ask({
      questions: [
        { id: 'plain', question: 'Proceed?', options: [{ label: 'Approve' }] },
        {
          id: 'plan-review', question: 'Approve?', detail: '# Plan',
          options: [{ label: 'Approve' }, { label: 'Keep planning' }], intent,
        },
      ],
    })

    expect(result.answers).toEqual([
      { id: 'plain', selected: ['Approve'] },
      { id: 'plan-review', selected: ['Approve'] },
    ])
    expect(p.seen[0]?.questions[1]?.intent).toEqual(intent)
  })

  it('hands the answerers a signal that spans the ask rather than the caller turn', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const watched: AbortSignal[] = []
    ctx.on('user-questions/request', (request, next) => {
      watched.push(request.signal as AbortSignal)
      return next()
    })
    const p = provider('yes')
    registerAnswerer(ctx, p)
    const caller = new AbortController()

    const result = await ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?', options: [{ label: 'yes' }] }],
      signal: caller.signal,
    })

    expect(result).toEqual({ answers: [{ id: 'confirm', selected: ['yes'] }] })
    expect(watched).toHaveLength(1)
    // The forwarded request never carries the caller's turn signal directly:
    // that would end every pending presentation when the TURN ends.
    expect(watched[0]).not.toBe(caller.signal)
    // The ask settled, so its own presentation lifetime ended with it.
    expect(watched[0]?.aborted).toBe(true)
    expect(caller.signal.aborted).toBe(false)
  })

  it('ends the delegated presentation when an earlier answerer claims the question', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    let delegated: AbortSignal | undefined
    // The competing surface runs first and races the delegation, as a
    // channel answerer does.
    ctx.on('user-questions/request', (_request, next) => Promise.race([
      Promise.resolve({ answers: [{ id: 'confirm', selected: ['claim'] }] }),
      next(),
    ]))
    // The forwarded presentation settles only when its human answers or the
    // presentation lifetime ends.
    ctx.on('user-questions/request', (request) => {
      delegated = request.signal
      return new Promise<AskUserQuestionAnswer>((_resolve, reject) => {
        const signal = request.signal
        if (signal === undefined) return
        if (signal.aborted) return reject(new Error('presentation cancelled'))
        signal.addEventListener('abort', () => { reject(new Error('presentation cancelled')) }, { once: true })
      })
    })

    const result = await ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?', options: [{ label: 'claim' }] }],
    })

    expect(result.answers).toEqual([{ id: 'confirm', selected: ['claim'] }])
    // The forwarded presentation ended with the ask, so it can no longer
    // collect a second answer for the same question.
    expect(delegated?.aborted).toBe(true)
  })

  it('keeps the caller signal as the cancellation authority over the ask lifetime', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    let watched: AbortSignal | undefined
    registerAnswerer(ctx, {
      ask(request) {
        watched = request.signal
        return new Promise<AskUserQuestionAnswer>((_resolve, reject) => {
          const signal = request.signal
          if (signal === undefined) return
          if (signal.aborted) return reject(new Error('presentation cancelled'))
          signal.addEventListener('abort', () => { reject(new Error('presentation cancelled')) }, { once: true })
        })
      },
    })
    const caller = new AbortController()
    const rejected = expect(ctx.userQuestions.ask({
      questions: [{ id: 'confirm', question: 'Proceed?' }],
      signal: caller.signal,
    })).rejects.toMatchObject({ name: 'UserQuestionError', code: 'ASK_ABORTED' })

    caller.abort(new Error('turn cancelled'))

    await rejected
    expect(watched?.aborted).toBe(true)
    expect(watched?.reason).toBe(caller.signal.reason)
  })

  it('gives every ask its own presentation lifetime', async () => {
    const ctx = new Context()
    await ctx.plugin(UserQuestionService)
    const watched: AbortSignal[] = []
    let previousAlreadyEnded = false
    ctx.on('user-questions/request', (request, next) => {
      watched.push(request.signal as AbortSignal)
      if (watched.length === 2) previousAlreadyEnded = watched[0]?.aborted === true
      return next()
    })
    registerAnswerer(ctx, provider('yes'))

    await ctx.userQuestions.ask({ questions: [{ id: 'q1', question: 'Proceed?' }] })
    await ctx.userQuestions.ask({ questions: [{ id: 'q2', question: 'Proceed?' }] })

    expect(watched).toHaveLength(2)
    expect(watched[0]).not.toBe(watched[1])
    // The earlier ask's lifetime had already ended before the next ask even
    // reached its answerers, so one ask never shares a lifetime with another.
    expect(previousAlreadyEnded).toBe(true)
  })
})
