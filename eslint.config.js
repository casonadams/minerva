import js from '@eslint/js';

export default [
  {
    ignores: ['node_modules/**', 'build/**', 'dist/**', '.svelte-kit/**'],
  },
  {
    files: ['src/**/*.{js,ts,mjs,mts,cjs,cts}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      globals: {
        browser: true,
        es2021: true,
        node: true,
      },
    },
    ...js.configs.recommended,
    rules: {
      'no-console': [
        'warn',
        {
          allow: ['warn', 'error'],
        },
      ],
      'no-debugger': 'warn',
      'no-var': 'error',
      'prefer-const': 'error',
      'prefer-arrow-callback': 'error',
      'arrow-body-style': ['error', 'as-needed'],
      'curly': ['error', 'all'],
      'brace-style': ['error', '1tbs'],
      'indent': ['error', 2, { SwitchCase: 1 }],
      'quotes': [
        'error',
        'single',
        {
          avoidEscape: true,
        },
      ],
      'semi': ['error', 'always'],
      'comma-dangle': ['error', 'always-multiline'],
      'object-curly-spacing': ['error', 'always'],
      'array-bracket-spacing': ['error', 'never'],
      'space-before-function-paren': [
        'error',
        {
          anonymous: 'always',
          named: 'never',
          asyncArrow: 'always',
        },
      ],
      'keyword-spacing': [
        'error',
        {
          before: true,
          after: true,
        },
      ],
      'space-infix-ops': 'error',
      'eqeqeq': ['error', 'always'],
      'no-implicit-coercion': 'error',
      'no-eval': 'error',
      'no-with': 'error',
      'no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
      'no-empty': [
        'error',
        {
          allowEmptyCatch: true,
        },
      ],
      'complexity': [
        'warn',
        {
          max: 10,
        },
      ],
      'max-len': [
        'warn',
        {
          code: 100,
          ignorePattern: '^import |^export ',
          ignoreUrls: true,
          ignoreStrings: true,
          ignoreTemplateLiterals: true,
        },
      ],
      'max-lines': [
        'warn',
        {
          max: 300,
          skipBlankLines: true,
          skipComments: true,
        },
      ],
      'max-params': [
        'warn',
        {
          max: 3,
        },
      ],
      'max-nested-callbacks': [
        'warn',
        {
          max: 3,
        },
      ],
      'max-depth': [
        'warn',
        {
          max: 4,
        },
      ],
    },
  },
];
