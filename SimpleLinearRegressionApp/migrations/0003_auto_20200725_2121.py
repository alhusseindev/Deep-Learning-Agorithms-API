# Generated by Django 3.0.6 on 2020-07-25 19:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('SimpleLinearRegressionApp', '0002_slrmodel_load_from_model'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='slrmodel',
            name='load_from_model',
        ),
        migrations.RemoveField(
            model_name='slrmodel',
            name='name',
        ),
        migrations.RemoveField(
            model_name='slrmodel',
            name='predictions',
        ),
        migrations.AddField(
            model_name='slrmodel',
            name='full_response',
            field=models.TextField(default=None),
        ),
        migrations.AddField(
            model_name='slrmodel',
            name='input_data',
            field=models.CharField(default=None, max_length=100000),
            preserve_default=False,
        ),
    ]
