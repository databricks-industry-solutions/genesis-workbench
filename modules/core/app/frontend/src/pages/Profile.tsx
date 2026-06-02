import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { Dialog } from '@/components/Dialog'
import { useUserStore } from '@/stores/user'

const ProfileFormSchema = z.object({
  user_display_name: z.string().min(1, 'Display name is required'),
  mlflow_experiment_folder: z.string().min(1, 'Folder name is required'),
})
type ProfileForm = z.infer<typeof ProfileFormSchema>

export function ProfilePage() {
  const bootstrap = useUserStore((s) => s.bootstrap)
  const setUserSettings = useUserStore((s) => s.setUserSettings)
  const qc = useQueryClient()

  const profileQuery = useQuery({ queryKey: ['profile'], queryFn: api.getProfile })

  const form = useForm<ProfileForm>({
    resolver: zodResolver(ProfileFormSchema),
    defaultValues: { user_display_name: '', mlflow_experiment_folder: 'mlflow_experiments' },
  })

  useEffect(() => {
    if (profileQuery.data) {
      const s = profileQuery.data.user_settings
      form.reset({
        user_display_name:
          s.user_display_name ||
          bootstrap?.user.display_name ||
          bootstrap?.user.preferred_username ||
          '',
        mlflow_experiment_folder: s.mlflow_experiment_folder || 'mlflow_experiments',
      })
    }
  }, [profileQuery.data, bootstrap, form])

  const save = useMutation({
    mutationFn: api.saveProfile,
    onSuccess: (data) => {
      setUserSettings(data.user_settings)
      qc.invalidateQueries({ queryKey: ['profile'] })
      qc.invalidateQueries({ queryKey: ['bootstrap'] })
    },
  })

  const onSubmit = (values: ProfileForm) => save.mutate(values)

  const appSpId = bootstrap?.env.app_service_principal_id
  const email = bootstrap?.user.email ?? ''
  const folderValue = form.watch('mlflow_experiment_folder')

  const [howTo, setHowTo] = useState<null | 'folder' | 'permissions'>(null)

  return (
    <div className="space-y-6 px-8 py-8">
      <p className="text-sm text-muted-foreground">
        One-time setup: pick a display name and an MLflow experiment folder that the app service
        principal can write to.
      </p>

      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-3 text-sm font-semibold">General</h2>
          <div className="space-y-3">
            <Field label="Email">
              <input
                value={email}
                disabled
                className="w-full rounded-md border border-border bg-muted px-3 py-2 text-sm text-muted-foreground"
              />
            </Field>
            <Field label="Display Name" error={form.formState.errors.user_display_name?.message}>
              <input
                {...form.register('user_display_name')}
                className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
              />
            </Field>
          </div>
        </section>

        <section className="rounded-lg border border-border bg-card p-5">
          <h2 className="mb-3 text-sm font-semibold">MLflow Setup</h2>
          <p className="mb-4 text-xs text-muted-foreground">
            Genesis Workbench creates per-user experiments to track work. Pick a folder under{' '}
            <code className="rounded bg-muted px-1">/Workspace/Users/{email}/</code> and grant the
            app service principal <strong>Can Manage</strong> permission on it.
          </p>

          <Field
            label="Folder Name"
            error={form.formState.errors.mlflow_experiment_folder?.message}
          >
            <input
              {...form.register('mlflow_experiment_folder')}
              className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm"
            />
          </Field>

          <div className="mt-4 space-y-3 text-xs text-muted-foreground">
            <div>
              <div className="flex items-baseline gap-2">
                <strong>Step 1.</strong>
                <span>
                  Create the folder if it doesn't exist:{' '}
                  <code className="rounded bg-muted px-1">
                    /Workspace/Users/{email}/{folderValue}
                  </code>
                </span>
              </div>
              <button
                type="button"
                onClick={() => setHowTo('folder')}
                className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary shadow-sm transition-colors hover:bg-primary/20"
              >
                <span className="material-symbols-outlined text-base leading-none">play_circle</span>
                Show me how
              </button>
            </div>
            <div>
              <div className="flex items-baseline gap-2">
                <strong>Step 2.</strong>
                <span>Share with the app service principal:</span>
              </div>
              <pre className="mt-1 overflow-auto rounded bg-muted p-2 text-xs">
                {appSpId ?? '(app service principal id not available)'}
              </pre>
              <button
                type="button"
                onClick={() => setHowTo('permissions')}
                className="mt-2 inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary shadow-sm transition-colors hover:bg-primary/20"
              >
                <span className="material-symbols-outlined text-base leading-none">play_circle</span>
                Show me how
              </button>
            </div>
            <div>
              <strong>Step 3.</strong> Grant <em>Can Manage</em> and submit. We'll create a test
              MLflow experiment in that folder to verify access.
            </div>
          </div>
        </section>

        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={save.isPending}
            className="rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:opacity-90 disabled:opacity-50"
          >
            {save.isPending ? 'Saving…' : 'Check Folder Permission and Save'}
          </button>
          {save.isSuccess && (
            <span className="text-sm text-success">Settings saved.</span>
          )}
          {save.error && (
            <span className="text-sm text-destructive">{String(save.error)}</span>
          )}
        </div>
      </form>

      <Dialog
        open={howTo !== null}
        onClose={() => setHowTo(null)}
        title={
          howTo === 'folder'
            ? 'How to create a workspace folder'
            : howTo === 'permissions'
              ? 'How to grant Can Manage permission'
              : ''
        }
        width="max-w-3xl"
      >
        {howTo === 'folder' && (
          <div className="space-y-3 text-sm">
            <p>
              Open the workspace browser at{' '}
              <code className="rounded bg-muted px-1">
                /Workspace/Users/{email || '<your email>'}/
              </code>
              , click <strong>Create → Folder</strong>, and type{' '}
              <code className="rounded bg-muted px-1">{folderValue}</code>.
            </p>
            <img
              src="/demo_new_folder.gif"
              alt="Create a new workspace folder"
              className="w-full rounded-md border border-border"
            />
          </div>
        )}
        {howTo === 'permissions' && (
          <div className="space-y-3 text-sm">
            <p>
              Navigate to the folder you just created, click the <strong>Share</strong> button on
              the top right, paste the app service principal id below, and pick{' '}
              <strong>Can Manage</strong>.
            </p>
            <pre className="overflow-auto rounded bg-muted p-2 text-xs">
              {appSpId ?? '(app service principal id not available)'}
            </pre>
            <img
              src="/set_permissions.gif"
              alt="Grant Can Manage permission to the service principal"
              className="w-full rounded-md border border-border"
            />
          </div>
        )}
      </Dialog>
    </div>
  )
}

function Field({
  label,
  error,
  children,
}: {
  label: string
  error?: string
  children: React.ReactNode
}) {
  return (
    <label className="block">
      <div className="mb-1 text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      {children}
      {error && <div className="mt-1 text-xs text-destructive">{error}</div>}
    </label>
  )
}
